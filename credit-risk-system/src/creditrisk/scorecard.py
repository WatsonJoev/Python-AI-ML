"""Traditional bank scorecard: WoE binning, Information Value, logistic regression, points per bin.

Conventions
-----------
* bad = default (y=1), good = non-default (y=0).
* WoE = ln(%good / %bad) per bin (with +0.5 smoothing), so a higher WoE means safer.
* IV  = sum((%good - %bad) * WoE).  Rule of thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, >0.3 strong.
* Score = offset + factor * ln(odds_good), factor = PDO / ln 2, offset = base_score - factor * ln(base_odds).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import ScoreCfg, ScorecardCfg

EPS = 1e-6


# --------------------------------------------------------------------------- score scaling
def score_factor_offset(cfg: ScoreCfg) -> tuple[float, float]:
    factor = cfg.pdo / math.log(2)
    offset = cfg.base_score - factor * math.log(cfg.base_odds)
    return factor, offset


def pd_to_score(pd_value: float | np.ndarray, cfg: ScoreCfg) -> float | np.ndarray:
    """Convert probability of default to a credit score (base 600 at 50:1 odds, 20 pts to double odds)."""
    factor, offset = score_factor_offset(cfg)
    p = np.clip(pd_value, EPS, 1 - EPS)
    return offset + factor * np.log((1 - p) / p)


def score_to_pd(score: float | np.ndarray, cfg: ScoreCfg) -> float | np.ndarray:
    factor, offset = score_factor_offset(cfg)
    odds_good = np.exp((np.asarray(score, dtype=float) - offset) / factor)
    return 1.0 / (1.0 + odds_good)


# --------------------------------------------------------------------------- binning
def _bin_labels(edges: np.ndarray) -> list[str]:
    if len(edges) == 0:
        return ["all"]
    labels = [f"<= {edges[0]:g}"]
    labels += [f"({edges[i]:g}, {edges[i + 1]:g}]" for i in range(len(edges) - 1)]
    labels.append(f"> {edges[-1]:g}")
    return labels


def make_edges(x: np.ndarray, max_bins: int, min_share: float) -> np.ndarray:
    """Quantile-based inner bin edges; bins are (e[i-1], e[i]]. Small bins are merged into a neighbour."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1]))
    n = len(x)
    while len(edges) > 0:
        idx = np.searchsorted(edges, x, side="left")
        shares = np.bincount(idx, minlength=len(edges) + 1) / n
        smallest = int(np.argmin(shares))
        if shares[smallest] >= min_share:
            break
        # drop the edge on the side of the smaller neighbour
        if smallest == 0:
            drop = 0
        elif smallest == len(edges):
            drop = len(edges) - 1
        else:
            drop = smallest - 1 if shares[smallest - 1] <= shares[smallest + 1] else smallest
        edges = np.delete(edges, drop)
    return edges


def assign_bins(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, x, side="left")


def woe_table(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    """Per-bin counts, bad rate, WoE and IV contribution."""
    n_bins = len(edges) + 1
    idx = assign_bins(x, edges)
    total_bad, total_good = float(y.sum()), float((1 - y).sum())
    rows = []
    for b, label in zip(range(n_bins), _bin_labels(edges)):
        mask = idx == b
        bad = float(y[mask].sum())
        good = float(mask.sum() - bad)
        pct_bad = (bad + 0.5) / (total_bad + 0.5 * n_bins)
        pct_good = (good + 0.5) / (total_good + 0.5 * n_bins)
        woe = math.log(pct_good / pct_bad)
        rows.append({"bin": label, "count": int(mask.sum()), "bad": int(bad), "good": int(good),
                     "bad_rate": bad / mask.sum() if mask.sum() else float("nan"),
                     "woe": woe, "iv": (pct_good - pct_bad) * woe})
    return pd.DataFrame(rows)


@dataclass
class WoeBinner:
    """Learns bin edges + WoE per feature on the training data."""
    cfg: ScorecardCfg
    edges: dict[str, np.ndarray] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoeBinner":
        yv = y.to_numpy(dtype=float)
        for col in X.columns:
            x = X[col].to_numpy(dtype=float)
            self.edges[col] = make_edges(x, self.cfg.max_bins, self.cfg.min_bin_share)
            self.tables[col] = woe_table(x, yv, self.edges[col])
        return self

    def transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        cols = columns or list(self.edges)
        out = {}
        for col in cols:
            idx = assign_bins(X[col].to_numpy(dtype=float), self.edges[col])
            out[col] = self.tables[col]["woe"].to_numpy()[idx]
        return pd.DataFrame(out, index=X.index)

    def bin_index(self, X: pd.DataFrame, col: str) -> np.ndarray:
        return assign_bins(X[col].to_numpy(dtype=float), self.edges[col])

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": c, "iv": float(t["iv"].sum()), "strength": iv_strength(float(t["iv"].sum()))}
                for c, t in self.tables.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "not useful"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspiciously strong"


# --------------------------------------------------------------------------- scorecard model
@dataclass
class Scorecard:
    """WoE-binned logistic regression with points per bin."""
    score_cfg: ScoreCfg
    cfg: ScorecardCfg
    binner: WoeBinner | None = None
    features: list[str] = field(default_factory=list)
    model: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self.binner = WoeBinner(self.cfg).fit(X, y)
        iv = self.binner.iv_table()
        self.features = iv.loc[iv["iv"] >= self.cfg.min_iv, "feature"].tolist()
        woe = self.binner.transform(X, self.features)
        # Unweighted: WoE inputs make the model well calibrated; imbalance is handled in evaluation/calibration.
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(woe, y)
        return self

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        assert self.binner is not None and self.model is not None
        return self.model.predict_proba(self.binner.transform(X, self.features))[:, 1]

    # ---- points
    def base_points(self) -> float:
        assert self.model is not None
        factor, offset = score_factor_offset(self.score_cfg)
        return float(offset - factor * self.model.intercept_[0])

    def points_table(self) -> pd.DataFrame:
        assert self.binner is not None and self.model is not None
        factor, _ = score_factor_offset(self.score_cfg)
        rows = []
        for feat, coef in zip(self.features, self.model.coef_[0]):
            for _, r in self.binner.tables[feat].iterrows():
                rows.append({"feature": feat, "bin": r["bin"], "woe": r["woe"], "bad_rate": r["bad_rate"],
                             "count": r["count"], "points": -factor * coef * r["woe"]})
        return pd.DataFrame(rows)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Score = base points + sum of bin points (equals pd_to_score of the raw scorecard PD)."""
        return self.base_points() + self._points_matrix(X).sum(axis=1)

    def _points_matrix(self, X: pd.DataFrame) -> np.ndarray:
        assert self.binner is not None and self.model is not None
        factor, _ = score_factor_offset(self.score_cfg)
        woe = self.binner.transform(X, self.features).to_numpy()
        return -factor * woe * self.model.coef_[0]

    def breakdown(self, X_row: pd.DataFrame) -> pd.DataFrame:
        """Points breakdown for ONE applicant (one-row frame)."""
        assert self.binner is not None
        pts = self._points_matrix(X_row)[0]
        rows = [{"item": "Base points", "bin": "", "points": self.base_points()}]
        for feat, p in zip(self.features, pts):
            b = int(self.binner.bin_index(X_row, feat)[0])
            rows.append({"item": feat, "bin": self.binner.tables[feat]["bin"].iloc[b], "points": float(p)})
        out = pd.DataFrame(rows)
        out.loc[len(out)] = {"item": "TOTAL SCORE", "bin": "", "points": out["points"].sum()}
        return out
