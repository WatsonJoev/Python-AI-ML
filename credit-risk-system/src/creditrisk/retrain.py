"""Retraining with feedback.

  python -m creditrisk.retrain                  merge recorded outcomes + training data, train a NEW versioned
                                                model, print a comparison with the active model on the SAME test set.
  python -m creditrisk.retrain --promote <ver>  explicitly make a version active (a deliberate human action).

The active model is NEVER replaced automatically.
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import store
from .config import Settings, get_settings, setup_logging
from .data import RAW_COLUMNS, TARGET, load_raw, split_data
from .features import api_payload_to_frame
from .scoring import active_version, load_active_artifact, save_artifact, set_active
from .train import evaluate_artifact, train_models

logger = logging.getLogger(__name__)
METRICS = ("roc_auc", "gini", "ks", "pr_auc", "brier")


def feedback_frame(settings: Settings) -> pd.DataFrame:
    """Recorded outcomes as a training-shaped frame (dataset columns + target)."""
    rows = store.feedback_rows(settings.paths.database)
    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS + [TARGET])
    frames = []
    for payload, defaulted in rows:
        f = api_payload_to_frame(payload)
        f[TARGET] = defaulted
        frames.append(f)
    return pd.concat(frames, ignore_index=True)[[TARGET] + RAW_COLUMNS]


def compare_table(active_ev: dict | None, new_ev: dict, active_ver: str | None, new_ver: str) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        row = {"metric": m, f"active ({active_ver})": active_ev["gradient_boosting"][m] if active_ev else float("nan"),
               f"candidate ({new_ver})": new_ev["gradient_boosting"][m]}
        rows.append(row)
    return pd.DataFrame(rows)


def run_retrain(settings: Settings, min_feedback: int = 1) -> int:
    feedback = feedback_frame(settings)
    if len(feedback) < min_feedback:
        logger.error("Only %d feedback rows (need >= %d). Record outcomes via POST /applications/{id}/outcome.",
                     len(feedback), min_feedback)
        return 1
    raw, source = load_raw(settings)
    splits = split_data(raw, settings)  # same seed => same held-out test set as the original training run
    artifact, _ = train_models(settings, raw, source, extra_train=feedback, splits=splits)
    new_ev = evaluate_artifact(settings, artifact, splits.test)
    artifact["test_metrics"] = {k: new_ev[k] for k in ("gradient_boosting", "scorecard")}

    active_ver = active_version(settings)
    active_ev = evaluate_artifact(settings, load_active_artifact(settings), splits.test) if active_ver else None
    path = save_artifact(settings, artifact)  # new file; refuses to overwrite
    print(f"\nData source: {source}; feedback rows merged into TRAIN only: {len(feedback)}")
    print("Comparison on the identical held-out test set (gradient boosting):")
    print(compare_table(active_ev, new_ev, active_ver, artifact["version"]).to_string(index=False, float_format="%.4f"))
    print(f"\nNew model written to {path}")
    print(f"The active model is UNCHANGED ({active_ver}). To switch after review:\n"
          f"  python -m creditrisk.retrain --promote {artifact['version']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--promote", metavar="VERSION", help="make this model version active (manual step)")
    ap.add_argument("--min-feedback", type=int, default=1, help="minimum outcome rows required to retrain")
    args = ap.parse_args(argv)
    settings = get_settings()
    if args.promote:
        set_active(settings, args.promote)
        print(f"Active model is now {args.promote}")
        return 0
    return run_retrain(settings, args.min_feedback)


if __name__ == "__main__":
    sys.exit(main())
