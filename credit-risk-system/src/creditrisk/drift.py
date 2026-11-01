"""Population Stability Index (PSI) monitoring of input features against the training data.

Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift (investigate), > 0.25 major shift (act).
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from .config import DriftCfg, get_settings, setup_logging
from .scorecard import assign_bins, make_edges

logger = logging.getLogger(__name__)
FLOOR = 1e-4  # avoids log(0) for empty bins


def psi_from_fractions(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), FLOOR, None)
    a = np.clip(np.asarray(actual, dtype=float), FLOOR, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _fractions(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(assign_bins(x, edges), minlength=len(edges) + 1)
    return counts / max(len(x), 1)


def psi_samples(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI between two samples of one variable, using quantile bins of the expected sample."""
    edges = make_edges(np.asarray(expected, dtype=float), n_bins, 0.0)
    return psi_from_fractions(_fractions(np.asarray(expected, dtype=float), edges),
                              _fractions(np.asarray(actual, dtype=float), edges))


def reference_profile(features: pd.DataFrame, n_bins: int = 10) -> dict[str, dict[str, list[float]]]:
    """Store compact bin edges + expected fractions from the training data (saved inside the model file)."""
    profile = {}
    for col in features.columns:
        x = features[col].to_numpy(dtype=float)
        edges = make_edges(x, n_bins, 0.0)
        profile[col] = {"edges": edges.tolist(), "fractions": _fractions(x, edges).tolist()}
    return profile


def status_for(value: float, cfg: DriftCfg) -> str:
    if value >= cfg.alert:
        return "ALERT (major shift)"
    if value >= cfg.warn:
        return "WARN (moderate shift)"
    return "OK (stable)"


def psi_report(profile: dict[str, dict[str, list[float]]], features: pd.DataFrame, cfg: DriftCfg) -> pd.DataFrame:
    rows = []
    for col, ref in profile.items():
        edges = np.asarray(ref["edges"], dtype=float)
        value = psi_from_fractions(np.asarray(ref["fractions"]), _fractions(features[col].to_numpy(dtype=float), edges))
        rows.append({"feature": col, "psi": value, "status": status_for(value, cfg)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Compare applications received by the service (SQLite) against the active model's training data."""
    from . import store
    from .features import api_payload_to_frame, prepare
    from .scoring import load_active_artifact

    setup_logging()
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    settings = get_settings()
    artifact = load_active_artifact(settings)
    payloads = store.all_application_payloads(settings.paths.database)
    if len(payloads) < settings.drift.min_rows:
        logger.error("Only %d stored applications; need at least %d for a meaningful PSI.",
                     len(payloads), settings.drift.min_rows)
        return 1
    from .data import CleaningParams
    raw = pd.concat([api_payload_to_frame(p) for p in payloads], ignore_index=True)
    feats = prepare(raw, CleaningParams(**artifact["cleaning_params"]), settings)
    print(psi_report(artifact["profile"], feats, settings.drift).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
