import pandas as pd
from fastapi.testclient import TestClient

from conftest import STRONG, make_settings
from creditrisk.api import create_app
from creditrisk.data import ALL_COLUMNS, SOURCE_KAGGLE, SOURCE_SYNTHETIC, generate_synthetic, load_raw


def test_loader_prefers_kaggle_file_when_present(tmp_path):
    s = make_settings(tmp_path)
    fake = generate_synthetic(500, 0.07, seed=5)
    fake.insert(0, "Unnamed: 0", range(1, len(fake) + 1))  # the real Kaggle CSV has this index column
    fake.to_csv(s.paths.kaggle_file, index=False)
    df, source = load_raw(s)
    assert source == SOURCE_KAGGLE and list(df.columns) == ALL_COLUMNS and len(df) == 500


def test_loader_falls_back_to_labelled_synthetic(tmp_path):
    df, source = load_raw(make_settings(tmp_path))
    assert source == SOURCE_SYNTHETIC and len(df) == 8000 and not pd.isna(df["SeriousDlqin2yrs"]).any()


def test_api_without_a_trained_model_returns_503_not_a_crash(tmp_path):
    client = TestClient(create_app(make_settings(tmp_path)))
    assert client.get("/health").json()["status"] == "degraded"
    assert client.post("/assess", json={"application_id": "Z1", **STRONG}).status_code == 503
