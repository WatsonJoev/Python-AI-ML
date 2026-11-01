import numpy as np
import pandas as pd

from creditrisk.config import CleaningCfg
from creditrisk.data import (AGE, DEBT_RATIO, DEPENDENTS, INCOME, OPEN_LINES, PD_30_59, PD_60_89, PD_90, REAL_ESTATE,
                             UTIL, CleaningParams, clean, generate_synthetic, split_data)

PARAMS = CleaningParams(income_median=5000.0, age_median=50.0)
CFG = CleaningCfg()


def row(**kw):
    base = {UTIL: 0.3, AGE: 40, PD_30_59: 0, DEBT_RATIO: 0.3, INCOME: 4000.0, OPEN_LINES: 5, PD_90: 0,
            REAL_ESTATE: 1, PD_60_89: 0, DEPENDENTS: 1.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_missing_income_imputed_with_train_median_and_flagged():
    out = clean(row(**{INCOME: np.nan}), PARAMS, CFG)
    assert out[INCOME].iloc[0] == 5000.0
    assert out["income_was_missing"].iloc[0] == 1


def test_present_income_not_flagged():
    assert clean(row(), PARAMS, CFG)["income_was_missing"].iloc[0] == 0


def test_missing_dependents_filled_with_configured_value():
    assert clean(row(**{DEPENDENTS: np.nan}), PARAMS, CFG)[DEPENDENTS].iloc[0] == CFG.dependents_fill


def test_outliers_are_capped():
    out = clean(row(**{UTIL: 4000.0, DEBT_RATIO: 3000.0, PD_30_59: 98, PD_60_89: 96, PD_90: 98,
                       INCOME: 9e6, OPEN_LINES: 80, REAL_ESTATE: 40, DEPENDENTS: 25.0}), PARAMS, CFG).iloc[0]
    assert out[UTIL] == CFG.utilization_cap
    assert out[DEBT_RATIO] == CFG.debt_ratio_cap
    assert out[PD_30_59] == out[PD_60_89] == out[PD_90] == CFG.past_due_cap
    assert out[INCOME] == CFG.monthly_income_cap
    assert out[OPEN_LINES] == CFG.open_credit_lines_cap
    assert out[REAL_ESTATE] == CFG.real_estate_loans_cap
    assert out[DEPENDENTS] == CFG.dependents_cap


def test_clean_does_not_mutate_input():
    df = row(**{UTIL: 99.0, INCOME: np.nan})
    before = df.copy()
    clean(df, PARAMS, CFG)
    pd.testing.assert_frame_equal(df, before)


def test_synthetic_data_has_expected_properties():
    df = generate_synthetic(20000, 0.07, seed=1)
    assert abs(df["SeriousDlqin2yrs"].mean() - 0.07) < 0.012
    assert 0.15 < df[INCOME].isna().mean() < 0.25
    assert df[UTIL].max() > 100 and (df[PD_30_59] >= 96).any()


def test_split_is_stratified_and_reproducible(settings, env):
    raw = env["raw"]
    a, b = split_data(raw, settings), split_data(raw, settings)
    pd.testing.assert_frame_equal(a.test, b.test)
    rates = [s["SeriousDlqin2yrs"].mean() for s in (a.train, a.validation, a.test)]
    assert max(rates) - min(rates) < 0.005
    assert len(a.train) + len(a.validation) + len(a.test) == len(raw)
