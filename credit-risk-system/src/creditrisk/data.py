"""Data loading, synthetic generation, cleaning and splitting.

Primary source: Kaggle "Give Me Some Credit" (cs-training.csv), placed manually in data/.
Fallback: a synthetic dataset with the same columns, so the project runs out of the box.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CleaningCfg, Settings

logger = logging.getLogger(__name__)

TARGET = "SeriousDlqin2yrs"
UTIL = "RevolvingUtilizationOfUnsecuredLines"
AGE = "age"
PD_30_59 = "NumberOfTime30-59DaysPastDueNotWorse"
DEBT_RATIO = "DebtRatio"
INCOME = "MonthlyIncome"
OPEN_LINES = "NumberOfOpenCreditLinesAndLoans"
PD_90 = "NumberOfTimes90DaysLate"
REAL_ESTATE = "NumberRealEstateLoansOrLines"
PD_60_89 = "NumberOfTime60-89DaysPastDueNotWorse"
DEPENDENTS = "NumberOfDependents"

RAW_COLUMNS = [UTIL, AGE, PD_30_59, DEBT_RATIO, INCOME, OPEN_LINES, PD_90, REAL_ESTATE, PD_60_89, DEPENDENTS]
ALL_COLUMNS = [TARGET] + RAW_COLUMNS

SOURCE_KAGGLE = "Kaggle 'Give Me Some Credit' (cs-training.csv)"
SOURCE_SYNTHETIC = "SYNTHETIC (generated, not real bank data)"


# --------------------------------------------------------------------------- generation
def generate_synthetic(n_rows: int, default_rate: float, seed: int) -> pd.DataFrame:
    """Generate a synthetic dataset with the Kaggle columns, missing income and outliers."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n_rows)  # latent "financial stress"

    age = np.clip(rng.normal(52, 15, n_rows), 21, 100).round()
    util = np.clip(rng.beta(1.0, 3.0, n_rows) + 0.12 * z, 0, 1.3)
    income = np.exp(rng.normal(8.6, 0.6, n_rows) - 0.1 * z)
    debt_ratio = np.clip(rng.beta(2, 5, n_rows) * 1.2 + 0.05 * z, 0, None)
    open_lines = rng.poisson(8, n_rows)
    real_estate = rng.poisson(1.0, n_rows)
    dependents = rng.poisson(0.8, n_rows).astype(float)
    pd_30 = rng.poisson(np.exp(-2.4 + 0.9 * z))
    pd_60 = rng.poisson(np.exp(-3.0 + 0.9 * z))
    pd_90 = rng.poisson(np.exp(-3.2 + 1.0 * z))

    logit_core = (
        0.9 * z + 1.4 * np.minimum(util, 1.2) + 0.25 * pd_30 + 0.35 * pd_60 + 0.45 * pd_90
        - 0.025 * (age - 50) + 0.6 * np.minimum(debt_ratio, 1.5) - 0.00004 * np.minimum(income, 20000)
    )
    lo, hi = -20.0, 5.0  # bisection on the intercept to hit the target default rate
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(mid + logit_core)))).mean()
        lo, hi = (mid, hi) if rate < default_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(lo + logit_core)))
    y = (rng.random(n_rows) < p).astype(int)

    # Realistic data problems: missing income / dependents and outliers
    income_missing = rng.random(n_rows) < 0.20
    debt_ratio = np.where(income_missing, rng.uniform(50, 5000, n_rows), debt_ratio)  # an amount, not a ratio
    income = np.where(income_missing, np.nan, income.round())
    dep_missing = rng.random(n_rows) < 0.026
    dependents = np.where(dep_missing, np.nan, dependents)
    outlier = rng.random(n_rows) < 0.01
    util = np.where(outlier, rng.uniform(2, 5000, n_rows), util)
    code = rng.random(n_rows) < 0.003  # Kaggle uses 96/98 as special codes in the past-due columns
    special = rng.choice([96, 98], n_rows)
    pd_30 = np.where(code, special, pd_30)
    pd_60 = np.where(code, special, pd_60)
    pd_90 = np.where(code, special, pd_90)

    return pd.DataFrame({
        TARGET: y, UTIL: util.round(6), AGE: age.astype(int), PD_30_59: pd_30, DEBT_RATIO: debt_ratio.round(6),
        INCOME: income, OPEN_LINES: open_lines, PD_90: pd_90, REAL_ESTATE: real_estate,
        PD_60_89: pd_60, DEPENDENTS: dependents,
    })


# --------------------------------------------------------------------------- loading
def load_raw(settings: Settings) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source label). Uses the Kaggle file when present, else synthetic data."""
    kaggle = Path(settings.paths.kaggle_file)
    if kaggle.exists():
        df = pd.read_csv(kaggle)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
        missing = set(ALL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{kaggle} is missing expected columns: {sorted(missing)}")
        logger.info("Loaded Kaggle dataset: %d rows from %s", len(df), kaggle)
        return df[ALL_COLUMNS].copy(), SOURCE_KAGGLE

    logger.warning("%s not found -> generating SYNTHETIC data (results are NOT real-world performance).", kaggle)
    df = generate_synthetic(settings.data.synthetic_rows, settings.data.synthetic_default_rate, settings.seed)
    out = Path(settings.paths.synthetic_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df, SOURCE_SYNTHETIC


# --------------------------------------------------------------------------- cleaning
@dataclass(frozen=True)
class CleaningParams:
    """Statistics learned on the TRAINING split only (avoids leakage) and reused at serving."""
    income_median: float
    age_median: float


def fit_cleaning_params(train: pd.DataFrame) -> CleaningParams:
    return CleaningParams(float(train[INCOME].median()), float(train[AGE].median()))


def clean(df: pd.DataFrame, params: CleaningParams, cfg: CleaningCfg) -> pd.DataFrame:
    """Apply the documented cleaning rules. Pure: returns a new frame, adds `income_was_missing`."""
    out = df.copy()
    out["income_was_missing"] = out[INCOME].isna().astype(int)
    out[INCOME] = out[INCOME].fillna(params.income_median).clip(lower=0, upper=cfg.monthly_income_cap)
    out[DEPENDENTS] = out[DEPENDENTS].fillna(cfg.dependents_fill).clip(0, cfg.dependents_cap)
    out[AGE] = out[AGE].fillna(params.age_median).clip(cfg.age_min, cfg.age_max)
    out[UTIL] = out[UTIL].clip(0, cfg.utilization_cap)
    out[DEBT_RATIO] = out[DEBT_RATIO].clip(0, cfg.debt_ratio_cap)
    for col in (PD_30_59, PD_60_89, PD_90):
        out[col] = out[col].clip(0, cfg.past_due_cap)
    out[OPEN_LINES] = out[OPEN_LINES].clip(0, cfg.open_credit_lines_cap)
    out[REAL_ESTATE] = out[REAL_ESTATE].clip(0, cfg.real_estate_loans_cap)
    return out


# --------------------------------------------------------------------------- splitting
@dataclass
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_data(df: pd.DataFrame, settings: Settings) -> Splits:
    """Stratified train / validation / test split with a fixed seed."""
    rest, test = train_test_split(df, test_size=settings.split.test_size, stratify=df[TARGET],
                                  random_state=settings.seed)
    val_share = settings.split.validation_size / (1 - settings.split.test_size)
    train, val = train_test_split(rest, test_size=val_share, stratify=rest[TARGET], random_state=settings.seed)
    return Splits(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))
