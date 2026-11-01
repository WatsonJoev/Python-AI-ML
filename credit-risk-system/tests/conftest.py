"""Shared fixtures: a tiny, fast, fully isolated environment (tmp paths, small synthetic data, small GB)."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from creditrisk.api import create_app
from creditrisk.config import Paths, Settings, load_settings
from creditrisk.data import load_raw
from creditrisk.scoring import Scorer
from creditrisk.train import train_models

STRONG = dict(revolving_utilization=0.05, age=45, past_due_30_59=0, debt_ratio=0.15, monthly_income=9000,
              open_credit_lines=8, past_due_90=0, real_estate_loans=1, past_due_60_89=0, dependents=1)
WEAK = dict(revolving_utilization=0.98, age=29, past_due_30_59=3, debt_ratio=0.90, monthly_income=2200,
            open_credit_lines=12, past_due_90=2, real_estate_loans=0, past_due_60_89=2, dependents=3)


def make_settings(root: Path) -> Settings:
    base = load_settings()
    paths = Paths(kaggle_file=root / "no_such_kaggle_file.csv", synthetic_file=root / "synthetic.csv",
                  models_dir=root / "models", reports_dir=root / "reports", database=root / "test.db")
    return base.model_copy(update={
        "paths": paths,
        "data": base.data.model_copy(update={"synthetic_rows": 8000}),
        "model": base.model.model_copy(update={"n_estimators": 40}),
    })


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    """Trained artifact + settings, built once per test session."""
    root = tmp_path_factory.mktemp("creditrisk")
    settings = make_settings(root)
    raw, source = load_raw(settings)
    artifact, splits = train_models(settings, raw, source)
    return {"settings": settings, "artifact": artifact, "raw": raw, "splits": splits, "root": root}


@pytest.fixture(scope="session")
def settings(env) -> Settings:
    return env["settings"]


@pytest.fixture(scope="session")
def artifact(env) -> dict:
    return env["artifact"]


def scorer_with_thresholds(settings: Settings, artifact: dict, thresholds: tuple[float, float]) -> Scorer:
    art = copy.copy(artifact)
    art["thresholds"] = thresholds
    return Scorer(settings, art)


@pytest.fixture()
def client_factory(settings, artifact, tmp_path):
    """Build a client with its own empty DB and (optionally) forced thresholds."""
    def build(thresholds: tuple[float, float] | None = None) -> TestClient:
        s = settings.model_copy(update={"paths": settings.paths.model_copy(update={"database": tmp_path / "api.db"})})
        scorer = scorer_with_thresholds(s, artifact, thresholds or artifact["thresholds"])
        return TestClient(create_app(s, scorer))
    return build


@pytest.fixture()
def client(client_factory) -> TestClient:
    return client_factory()
