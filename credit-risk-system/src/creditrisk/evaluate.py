"""Metrics, calibration, cost-based decision thresholds and the age-group fairness check."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score, roc_curve)

from .config import CostsCfg, RiskBandsCfg, ThresholdsCfg

APPROVE, REFER, DECLINE = "APPROVE", "REFER", "DECLINE"
EPS = 1e-6


# --------------------------------------------------------------------------- metrics
def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between the score CDFs of bads and goods."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y, p))
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "default_rate": float(np.mean(y)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate per PD decile (a well-calibrated model has them equal)."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    t = df.groupby("bucket").agg(count=("y", "size"), mean_predicted_pd=("p", "mean"),
                                 observed_default_rate=("y", "mean")).reset_index()
    t["abs_gap"] = (t["mean_predicted_pd"] - t["observed_default_rate"]).abs()
    return t


# --------------------------------------------------------------------------- calibration
@dataclass
class PlattCalibrator:
    """Platt scaling: logistic regression on logit(raw score). Restores true PDs after class re-weighting."""
    lr: LogisticRegression | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, EPS, 1 - EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p_raw), y)
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        assert self.lr is not None
        return self.lr.predict_proba(self._logit(np.asarray(p_raw, dtype=float)))[:, 1]


# --------------------------------------------------------------------------- thresholds
def total_cost(y: np.ndarray, p: np.ndarray, decline_at: float, costs: CostsCfg) -> float:
    """Cost of a single cut-off: approved defaulters cost `cost_of_default`, declined goods cost lost profit."""
    declined = p >= decline_at
    missed_bad = np.sum((y == 1) & ~declined)
    lost_good = np.sum((y == 0) & declined)
    return float(missed_bad * costs.cost_of_default + lost_good * costs.lost_profit_good)


def breakeven_pd(costs: CostsCfg) -> float:
    """Analytic cut-off for a perfectly calibrated model: decline when PD*C > (1-PD)*P  =>  PD > P/(C+P)."""
    return costs.lost_profit_good / (costs.cost_of_default + costs.lost_profit_good)


def select_thresholds(y: np.ndarray, p: np.ndarray, costs: CostsCfg,
                      override: ThresholdsCfg | None = None) -> tuple[float, float]:
    """Return (approve_below, decline_at_or_above).

    Decline cut-off = PD that minimises total expected cost on validation data.
    Approve cut-off = refer_ratio * decline cut-off; PDs in between are referred to a human.
    Explicit overrides in config win.
    """
    grid = np.linspace(0.005, 0.60, 240)
    costs_by_t = np.array([total_cost(y, p, t, costs) for t in grid])
    decline = float(grid[int(np.argmin(costs_by_t))])
    approve = decline * costs.refer_ratio
    if override is not None:
        decline = override.decline_at_or_above if override.decline_at_or_above is not None else decline
        approve = override.approve_below if override.approve_below is not None else approve
    if approve > decline:
        raise ValueError("approve_below must not exceed decline_at_or_above")
    return float(approve), float(decline)


def decide(p: float | np.ndarray, approve_below: float, decline_at: float) -> np.ndarray | str:
    p_arr = np.asarray(p, dtype=float)
    out = np.where(p_arr >= decline_at, DECLINE, np.where(p_arr < approve_below, APPROVE, REFER))
    return str(out) if out.ndim == 0 else out


def risk_band(p: float, cfg: RiskBandsCfg) -> str:
    if p < cfg.low_below:
        return "Low"
    if p >= cfg.high_at_or_above:
        return "High"
    return "Medium"


def decision_summary(y: np.ndarray, p: np.ndarray, approve_below: float, decline_at: float,
                     costs: CostsCfg) -> dict[str, float]:
    d = decide(p, approve_below, decline_at)
    approved = d == APPROVE
    cost_model = total_cost(y, p, decline_at, costs)
    cost_approve_all = float(np.sum(y == 1) * costs.cost_of_default)
    return {
        "approve_rate": float(np.mean(d == APPROVE)),
        "refer_rate": float(np.mean(d == REFER)),
        "decline_rate": float(np.mean(d == DECLINE)),
        "default_rate_among_approved": float(y[approved].mean()) if approved.any() else float("nan"),
        "default_rate_among_declined": float(y[d == DECLINE].mean()) if (d == DECLINE).any() else float("nan"),
        "expected_cost_model": cost_model,
        "expected_cost_approve_everyone": cost_approve_all,
        "cost_saving_vs_approve_everyone": cost_approve_all - cost_model,
    }


# --------------------------------------------------------------------------- fairness
AGE_BINS = [0, 24, 34, 49, 64, 200]
AGE_LABELS = ["<25", "25-34", "35-49", "50-64", "65+"]


def age_group_report(age: pd.Series, y: np.ndarray, p: np.ndarray, approve_below: float,
                     decline_at: float) -> pd.DataFrame:
    """Performance and outcomes by age group. Age is NOT a model input; this is a monitoring check."""
    groups = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS)
    d = decide(p, approve_below, decline_at)
    rows = []
    for g in AGE_LABELS:
        m = (groups == g).to_numpy()
        if m.sum() == 0:
            continue
        auc = float(roc_auc_score(y[m], p[m])) if len(np.unique(y[m])) == 2 else float("nan")
        rows.append({"age_group": g, "count": int(m.sum()), "actual_default_rate": float(y[m].mean()),
                     "mean_predicted_pd": float(p[m].mean()), "roc_auc": auc,
                     "approve_rate": float(np.mean(d[m] == APPROVE)), "decline_rate": float(np.mean(d[m] == DECLINE))})
    return pd.DataFrame(rows)
