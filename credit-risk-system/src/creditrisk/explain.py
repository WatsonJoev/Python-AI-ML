"""Explainability: SHAP values for the gradient boosting model -> plain-English reason codes.

Only features that INCREASE risk (positive SHAP, log-odds scale) are returned as decline/refer reasons.
For an approval we return the strongest risk-REDUCING factors instead.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import shap

from .data import DEBT_RATIO, DEPENDENTS, INCOME, OPEN_LINES, PD_30_59, PD_60_89, PD_90, REAL_ESTATE, UTIL
from .evaluate import APPROVE
from .features import (HIGH_DEBT_FLAG, INCOME_PER_DEPENDENT, MISSING_INCOME_FLAG, MODEL_FEATURES, TOTAL_PAST_DUE)

MIN_CONTRIBUTION = 1e-3  # ignore negligible SHAP values
# A zero here is the "good" state; stating it as a RISK reason would be misleading (e.g. "no late payments").
ZERO_IS_GOOD = {PD_30_59, PD_60_89, PD_90, TOTAL_PAST_DUE, HIGH_DEBT_FLAG, MISSING_INCOME_FLAG}


def _level(value: float, levels: list[tuple[float, str]], top: str) -> str:
    for upper, word in levels:
        if value < upper:
            return word
    return top


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _leveled(label: str, fmt: Callable[[float], str], levels: list[tuple[float, str]], top: str
             ) -> Callable[[float], str]:
    return lambda v: f"{label} is {_level(v, levels, top)} ({fmt(v)})"


def _count_late(span: str) -> Callable[[float], str]:
    def f(v: float) -> str:
        n = int(round(v))
        if n == 0:
            return f"No payments {span} late in the last 2 years"
        return f"{n} payment{'s' if n != 1 else ''} {span} late in the last 2 years"
    return f


REASON_TEXT: dict[str, Callable[[float], str]] = {
    UTIL: _leveled("Revolving credit utilization", _pct, [(0.3, "low"), (0.6, "moderate"), (0.9, "high")], "very high"),
    DEBT_RATIO: _leveled("Debt-to-income ratio", _pct, [(0.2, "low"), (0.43, "moderate"), (1.0, "high")], "very high"),
    INCOME: _leveled("Monthly income", lambda v: f"${v:,.0f}", [(2500, "low"), (5000, "moderate"), (10000, "good")],
                     "high"),
    OPEN_LINES: _leveled("Number of open credit lines and loans", lambda v: f"{v:.0f}", [(4, "low"), (11, "moderate")],
                         "high"),
    REAL_ESTATE: _leveled("Number of real estate loans", lambda v: f"{v:.0f}", [(2, "low"), (4, "moderate")], "high"),
    DEPENDENTS: _leveled("Number of dependents", lambda v: f"{v:.0f}", [(2, "low"), (4, "moderate")], "high"),
    PD_30_59: _count_late("30-59 days"),
    PD_60_89: _count_late("60-89 days"),
    PD_90: _count_late("90+ days"),
    TOTAL_PAST_DUE: lambda v: ("No past-due payments on record" if round(v) == 0 else
                               f"Total number of late payments (all severities) is {int(round(v))}"),
    INCOME_PER_DEPENDENT: _leveled("Income per household member", lambda v: f"${v:,.0f}/month",
                                   [(1500, "low"), (3000, "moderate")], "comfortable"),
    HIGH_DEBT_FLAG: lambda v: ("Debt payments are a high share of income (above the 43% guideline)" if v >= 0.5
                               else "Debt-to-income ratio is within the 43% guideline"),
    MISSING_INCOME_FLAG: lambda v: ("Income was not provided or could not be verified" if v >= 0.5
                                    else "Income was provided"),
}
assert set(REASON_TEXT) == set(MODEL_FEATURES), "every model feature needs a reason template"


def make_explainer(gb_model: object) -> shap.TreeExplainer:
    return shap.TreeExplainer(gb_model)


def shap_matrix(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (n_rows, n_features) in log-odds of the positive (default) class."""
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def reasons_for_row(shap_row: np.ndarray, feature_row: pd.Series, decision: str, max_reasons: int = 4
                    ) -> list[str]:
    """Plain-English reasons from one row of SHAP values."""
    order = np.argsort(-shap_row) if decision != APPROVE else np.argsort(shap_row)
    reasons: list[str] = []
    for i in order:
        contribution = shap_row[i] if decision != APPROVE else -shap_row[i]
        if contribution < MIN_CONTRIBUTION or len(reasons) >= max_reasons:
            break
        feat = MODEL_FEATURES[i]
        if decision != APPROVE and feat in ZERO_IS_GOOD and float(feature_row[feat]) < 0.5:
            continue
        reasons.append(REASON_TEXT[feat](float(feature_row[feat])))
    if not reasons:
        reasons.append("No single factor materially increased risk" if decision != APPROVE
                       else "No single factor strongly reduced risk")
    return reasons
