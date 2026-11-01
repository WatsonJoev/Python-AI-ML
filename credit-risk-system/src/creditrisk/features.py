"""Feature engineering. The SAME function is used for training and serving (no train/serve skew).

`prepare(raw_df, params, settings)` = clean -> engineer -> select model columns.
Age is deliberately NOT a model feature (fair-lending); it is only used for the fairness report.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .data import (AGE, DEBT_RATIO, DEPENDENTS, INCOME, OPEN_LINES, PD_30_59, PD_60_89, PD_90, RAW_COLUMNS,
                   REAL_ESTATE, UTIL, CleaningParams, clean)

TOTAL_PAST_DUE = "TotalPastDue"
INCOME_PER_DEPENDENT = "IncomePerDependent"
HIGH_DEBT_FLAG = "HighDebtRatioFlag"
MISSING_INCOME_FLAG = "MissingIncomeFlag"

MODEL_FEATURES: list[str] = [
    UTIL, DEBT_RATIO, INCOME, OPEN_LINES, REAL_ESTATE, DEPENDENTS,
    PD_30_59, PD_60_89, PD_90,
    TOTAL_PAST_DUE, INCOME_PER_DEPENDENT, HIGH_DEBT_FLAG, MISSING_INCOME_FLAG,
]
assert AGE not in MODEL_FEATURES

# API (snake_case) field -> dataset column
API_TO_COLUMN: dict[str, str] = {
    "revolving_utilization": UTIL,
    "age": AGE,
    "past_due_30_59": PD_30_59,
    "debt_ratio": DEBT_RATIO,
    "monthly_income": INCOME,
    "open_credit_lines": OPEN_LINES,
    "past_due_90": PD_90,
    "real_estate_loans": REAL_ESTATE,
    "past_due_60_89": PD_60_89,
    "dependents": DEPENDENTS,
}


def engineer(cleaned: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add engineered features to a CLEANED frame."""
    out = cleaned.copy()
    out[TOTAL_PAST_DUE] = out[PD_30_59] + out[PD_60_89] + out[PD_90]
    out[INCOME_PER_DEPENDENT] = out[INCOME] / (out[DEPENDENTS] + 1)
    out[HIGH_DEBT_FLAG] = (out[DEBT_RATIO] > settings.features.high_debt_ratio_flag).astype(int)
    out[MISSING_INCOME_FLAG] = out["income_was_missing"]
    return out


def prepare(raw: pd.DataFrame, params: CleaningParams, settings: Settings) -> pd.DataFrame:
    """Raw applicant rows -> model-ready feature matrix. Used by training AND serving."""
    cleaned = clean(raw, params, settings.cleaning)
    return engineer(cleaned, settings)[MODEL_FEATURES].astype(float)


def api_payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert one API-style applicant dict into a one-row raw frame with dataset columns."""
    row = {col: payload.get(api_name) for api_name, col in API_TO_COLUMN.items()}
    row = {k: (np.nan if v is None else v) for k, v in row.items()}
    return pd.DataFrame([row], columns=RAW_COLUMNS).astype(float)
