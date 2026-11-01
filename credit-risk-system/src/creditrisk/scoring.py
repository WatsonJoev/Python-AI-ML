"""Model registry + the Scorer used by the API and the demo: applicant -> PD, score, band, decision, reasons."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .data import CleaningParams
from .evaluate import decide, risk_band
from .explain import make_explainer, reasons_for_row, shap_matrix
from .features import api_payload_to_frame, prepare
from .scorecard import pd_to_score

logger = logging.getLogger(__name__)
ACTIVE_POINTER = "ACTIVE_MODEL"


class NoActiveModelError(RuntimeError):
    """No trained model is active. Run `python -m creditrisk.train` first."""


# --------------------------------------------------------------------------- registry
def model_path(settings: Settings, version: str) -> Path:
    return Path(settings.paths.models_dir) / f"model_{version}.joblib"


def save_artifact(settings: Settings, artifact: dict[str, Any]) -> Path:
    path = model_path(settings, artifact["version"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} already exists; model files are never overwritten")
    joblib.dump(artifact, path)
    return path


def active_version(settings: Settings) -> str | None:
    pointer = Path(settings.paths.models_dir) / ACTIVE_POINTER
    return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None


def set_active(settings: Settings, version: str) -> None:
    """Explicitly promote a version. Only called by train (first model) or `retrain --promote`."""
    if not model_path(settings, version).exists():
        raise FileNotFoundError(model_path(settings, version))
    (Path(settings.paths.models_dir) / ACTIVE_POINTER).write_text(version, encoding="utf-8")


def load_artifact(settings: Settings, version: str) -> dict[str, Any]:
    # joblib files are pickles: only load files that this project wrote itself.
    return joblib.load(model_path(settings, version))


def load_active_artifact(settings: Settings) -> dict[str, Any]:
    version = active_version(settings)
    if version is None:
        raise NoActiveModelError("No active model. Run `python -m creditrisk.train` first.")
    return load_artifact(settings, version)


# --------------------------------------------------------------------------- scoring
@dataclass(frozen=True)
class Assessment:
    pd: float
    credit_score: int
    risk_band: str
    decision: str
    reasons: list[str]
    model_version: str


class Scorer:
    """Wraps one model artifact. Uses the SAME `prepare` function as training."""

    def __init__(self, settings: Settings, artifact: dict[str, Any]) -> None:
        self.settings = settings
        self.artifact = artifact
        self.version: str = artifact["version"]
        self.params = CleaningParams(**artifact["cleaning_params"])
        self.approve_below, self.decline_at = artifact["thresholds"]
        self._explainer = make_explainer(artifact["gb"])

    @classmethod
    def from_active(cls, settings: Settings) -> "Scorer":
        return cls(settings, load_active_artifact(settings))

    def predict_pd(self, features: pd.DataFrame) -> np.ndarray:
        return self.artifact["gb_calibrator"].predict(self.artifact["gb"].predict_proba(features)[:, 1])

    def assess_frame(self, raw: pd.DataFrame) -> list[Assessment]:
        feats = prepare(raw, self.params, self.settings)
        pds = self.predict_pd(feats)
        shap_vals = shap_matrix(self._explainer, feats)
        out = []
        for i, p in enumerate(pds):
            decision = str(decide(p, self.approve_below, self.decline_at))
            reasons = reasons_for_row(shap_vals[i], feats.iloc[i], decision, self.settings.reasons.max_reasons)
            out.append(Assessment(float(p), int(round(float(pd_to_score(p, self.settings.score)))),
                                  risk_band(float(p), self.settings.risk_bands), decision, reasons, self.version))
        return out

    def assess(self, applicant: dict[str, Any]) -> Assessment:
        """Score one applicant given as an API-style dict (snake_case keys)."""
        return self.assess_frame(api_payload_to_frame(applicant))[0]
