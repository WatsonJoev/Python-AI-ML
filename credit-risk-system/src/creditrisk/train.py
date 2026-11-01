"""Training pipeline: clean -> features -> scorecard + gradient boosting -> calibrate -> thresholds -> report.

Run:  python -m creditrisk.train
Class imbalance (~7% defaults) is handled by:
  1. stratified splits (same default rate in train / validation / test),
  2. balanced sample weights for the gradient boosting model,
  3. Platt calibration on the validation set, which restores true PDs after re-weighting,
  4. imbalance-robust metrics (ROC-AUC, Gini, KS, PR-AUC, Brier) instead of accuracy.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_sample_weight

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import Settings, get_settings, setup_logging  # noqa: E402
from .data import AGE, TARGET, CleaningParams, Splits, clean, fit_cleaning_params, load_raw, split_data  # noqa: E402
from .drift import psi_report, reference_profile  # noqa: E402
from .evaluate import (PlattCalibrator, age_group_report, calibration_table, compute_metrics,  # noqa: E402
                       decision_summary, select_thresholds)
from .features import MODEL_FEATURES, prepare  # noqa: E402
from .scorecard import Scorecard  # noqa: E402
from .scoring import active_version, save_artifact, set_active  # noqa: E402

logger = logging.getLogger(__name__)


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def fit_gradient_boosting(settings: Settings, X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
    m = settings.model
    gb = GradientBoostingClassifier(n_estimators=m.n_estimators, learning_rate=m.learning_rate,
                                    max_depth=m.max_depth, subsample=m.subsample,
                                    min_samples_leaf=m.min_samples_leaf, random_state=settings.seed)
    return gb.fit(X, y, sample_weight=compute_sample_weight("balanced", y))


def evaluate_artifact(settings: Settings, artifact: dict[str, Any], test_raw: pd.DataFrame) -> dict[str, Any]:
    """Score an artifact on a held-out raw test frame. Used by train AND retrain (same test set)."""
    params = CleaningParams(**artifact["cleaning_params"])
    X = prepare(test_raw, params, settings)
    y = test_raw[TARGET].to_numpy()
    p_gb = artifact["gb_calibrator"].predict(artifact["gb"].predict_proba(X)[:, 1])
    p_sc = artifact["scorecard_calibrator"].predict(artifact["scorecard"].predict_proba_raw(X))
    approve, decline = artifact["thresholds"]
    return {
        "gradient_boosting": compute_metrics(y, p_gb),
        "scorecard": compute_metrics(y, p_sc),
        "decisions_gradient_boosting": decision_summary(y, p_gb, approve, decline, settings.costs),
        "_p_gb": p_gb, "_p_sc": p_sc, "_y": y,
    }


def train_models(settings: Settings, raw: pd.DataFrame, source: str, extra_train: pd.DataFrame | None = None,
                 splits: Splits | None = None) -> tuple[dict[str, Any], Splits]:
    """Fit everything and return (artifact, splits). `extra_train` (feedback rows) is added to TRAIN only."""
    splits = splits or split_data(raw, settings)
    train_raw = splits.train if extra_train is None else pd.concat([splits.train, extra_train], ignore_index=True)

    params = fit_cleaning_params(train_raw)  # learned on train only -> no leakage
    X_tr, y_tr = prepare(train_raw, params, settings), train_raw[TARGET]
    X_va, y_va = prepare(splits.validation, params, settings), splits.validation[TARGET].to_numpy()

    logger.info("Fitting scorecard (WoE + logistic regression) on %d rows", len(X_tr))
    scorecard = Scorecard(settings.score, settings.scorecard).fit(X_tr, y_tr)
    logger.info("Fitting gradient boosting on %d rows", len(X_tr))
    gb = fit_gradient_boosting(settings, X_tr, y_tr)

    gb_cal = PlattCalibrator().fit(gb.predict_proba(X_va)[:, 1], y_va)
    sc_cal = PlattCalibrator().fit(scorecard.predict_proba_raw(X_va), y_va)
    p_val = gb_cal.predict(gb.predict_proba(X_va)[:, 1])
    thresholds = select_thresholds(y_va, p_val, settings.costs, settings.thresholds)

    artifact: dict[str, Any] = {
        "version": new_version(), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source, "n_train": int(len(train_raw)), "n_feedback_rows": 0 if extra_train is None else len(extra_train),
        "train_default_rate": float(y_tr.mean()), "cleaning_params": {"income_median": params.income_median,
                                                                      "age_median": params.age_median},
        "gb": gb, "gb_calibrator": gb_cal, "scorecard": scorecard, "scorecard_calibrator": sc_cal,
        "thresholds": thresholds, "feature_names": MODEL_FEATURES,
        "profile": reference_profile(X_tr),
    }
    return artifact, splits


def write_reports(settings: Settings, artifact: dict[str, Any], splits: Splits, ev: dict[str, Any]) -> dict[str, Any]:
    """Write CSV/PNG/JSON reports under reports/ and return the metrics summary."""
    out = Path(settings.paths.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc: Scorecard = artifact["scorecard"]
    y, p_gb, p_sc = ev["_y"], ev["_p_gb"], ev["_p_sc"]
    approve, decline = artifact["thresholds"]

    sc.binner.iv_table().to_csv(out / "iv_table.csv", index=False)
    sc.points_table().to_csv(out / "scorecard_points.csv", index=False)
    calibration_table(y, p_gb).to_csv(out / "calibration_gb.csv", index=False)
    calibration_table(y, p_sc).to_csv(out / "calibration_scorecard.csv", index=False)
    age_report = age_group_report(clean(splits.test, CleaningParams(**artifact["cleaning_params"]),
                                        settings.cleaning)[AGE], y, p_gb, approve, decline)
    age_report.to_csv(out / "fairness_by_age_group.csv", index=False)

    params = CleaningParams(**artifact["cleaning_params"])
    X_te = prepare(splits.test, params, settings)
    psi_report(artifact["profile"], X_te, settings.drift).to_csv(out / "psi_test_vs_train.csv", index=False)
    shifted = X_te.copy()  # illustrative drift scenario: a weaker, more stretched applicant pool
    shifted["RevolvingUtilizationOfUnsecuredLines"] = (shifted["RevolvingUtilizationOfUnsecuredLines"] * 1.5).clip(upper=1.5)
    shifted["MonthlyIncome"] = shifted["MonthlyIncome"] * 0.7
    psi_report(artifact["profile"], shifted, settings.drift).to_csv(out / "psi_simulated_shift.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, key, p in (("Gradient boosting", "gradient_boosting", p_gb), ("Scorecard", "scorecard", p_sc)):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {ev[key]['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC (test set)", xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend()
    for name, p in (("Gradient boosting", p_gb), ("Scorecard", p_sc)):
        t = calibration_table(y, p)
        axes[1].plot(t["mean_predicted_pd"], t["observed_default_rate"], marker="o", label=name)
    top = float(max(p_gb.max(), p_sc.max()))
    axes[1].plot([0, top], [0, top], "k--", lw=0.8)
    axes[1].set(title="Calibration by PD decile (test set)", xlabel="Mean predicted PD", ylabel="Observed default rate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "roc_and_calibration.png", dpi=130)
    plt.close(fig)

    summary = {
        "version": artifact["version"], "data_source": artifact["data_source"],
        "rows": {"train": int(len(splits.train)), "validation": int(len(splits.validation)), "test": int(len(splits.test))},
        "train_default_rate": artifact["train_default_rate"],
        "thresholds": {"approve_below": approve, "decline_at_or_above": decline},
        "test_metrics": {"gradient_boosting": ev["gradient_boosting"], "scorecard": ev["scorecard"]},
        "decisions_gradient_boosting": ev["decisions_gradient_boosting"],
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"\nData source : {summary['data_source']}")
    print(f"Model       : {summary['version']}   rows: {summary['rows']}")
    t = summary["thresholds"]
    print(f"Thresholds  : APPROVE if PD < {t['approve_below']:.4f}; DECLINE if PD >= {t['decline_at_or_above']:.4f}; else REFER")
    tm = summary["test_metrics"]
    print("\nTEST SET METRICS (same held-out rows for both models)")
    print(f"{'metric':<10}{'scorecard':>12}{'grad.boost':>12}")
    for k in ("roc_auc", "gini", "ks", "pr_auc", "brier"):
        print(f"{k:<10}{tm['scorecard'][k]:>12.4f}{tm['gradient_boosting'][k]:>12.4f}")
    print("\nDecisions (gradient boosting) on the test set")
    for k, v in summary["decisions_gradient_boosting"].items():
        print(f"  {k:<34}{v:>14,.4f}")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    argparse.ArgumentParser(description="Train the CreditRisk models.").parse_args(argv)
    settings = get_settings()
    raw, source = load_raw(settings)
    logger.info("Data source: %s (%d rows, default rate %.2f%%)", source, len(raw), 100 * raw[TARGET].mean())
    artifact, splits = train_models(settings, raw, source)
    ev = evaluate_artifact(settings, artifact, splits.test)
    artifact["test_metrics"] = {k: ev[k] for k in ("gradient_boosting", "scorecard")}
    path = save_artifact(settings, artifact)
    summary = write_reports(settings, artifact, splits, ev)
    if active_version(settings) is None:
        set_active(settings, artifact["version"])
        logger.info("No active model existed -> %s set as active.", artifact["version"])
    else:
        logger.info("Active model left unchanged (%s). Promote explicitly with retrain --promote.", active_version(settings))
    print_summary(summary)
    print(f"\nSaved model: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
