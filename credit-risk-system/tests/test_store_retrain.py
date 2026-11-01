import sqlite3
import time

import pytest

from conftest import STRONG
from creditrisk import store
from creditrisk.retrain import run_retrain
from creditrisk.scoring import active_version, save_artifact, set_active


def test_application_and_decision_written_atomically(tmp_path):
    db = tmp_path / "t.db"
    store.init_db(db)
    store.save_assessment(db, "T1", STRONG, 0.02, 650, "Low", "APPROVE", ["ok"], "v1")
    with pytest.raises(store.DuplicateApplicationError):
        store.save_assessment(db, "T1", STRONG, 0.02, 650, "Low", "APPROVE", ["ok"], "v1")
    # a decision violating a constraint must roll back the application insert too
    with pytest.raises(sqlite3.IntegrityError):
        store.save_assessment(db, "T2", STRONG, 0.02, 650, "Low", "MAYBE", ["x"], "v1")
    assert not store.application_exists(db, "T2")


def test_retrain_writes_new_version_and_keeps_active_model(env, settings, tmp_path):
    s = settings.model_copy(update={"paths": settings.paths.model_copy(update={
        "models_dir": tmp_path / "models", "database": tmp_path / "fb.db"})})
    save_artifact(s, env["artifact"])
    set_active(s, env["artifact"]["version"])
    store.init_db(s.paths.database)
    for i in range(5):
        store.save_assessment(s.paths.database, f"F{i}", STRONG, 0.02, 650, "Low", "APPROVE", ["ok"], "v1")
        store.record_outcome(s.paths.database, f"F{i}", defaulted=bool(i % 2))
    before = active_version(s)
    time.sleep(1.1)  # versions are second-resolution timestamps
    assert run_retrain(s) == 0
    assert len(list((tmp_path / "models").glob("model_*.joblib"))) == 2  # old + NEW file, nothing overwritten
    assert active_version(s) == before                                   # active model NOT switched


def test_retrain_without_feedback_refuses(settings, tmp_path):
    s = settings.model_copy(update={"paths": settings.paths.model_copy(update={"database": tmp_path / "empty.db"})})
    assert run_retrain(s) == 1
