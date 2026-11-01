import numpy as np
import pytest

from creditrisk.config import CostsCfg, DriftCfg, RiskBandsCfg, ThresholdsCfg
from creditrisk.drift import psi_samples, status_for
from creditrisk.evaluate import (APPROVE, DECLINE, REFER, PlattCalibrator, breakeven_pd, calibration_table, decide,
                                 ks_statistic, risk_band, select_thresholds, total_cost)

COSTS = CostsCfg(cost_of_default=10000, lost_profit_good=1000, refer_ratio=0.4)


def calibrated_sample(n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(1, 12, n), 0.001, 0.9)
    return (rng.random(n) < p).astype(int), p


def test_breakeven_formula():
    assert breakeven_pd(COSTS) == pytest.approx(1000 / 11000)


def test_selected_threshold_close_to_breakeven_for_calibrated_pds():
    y, p = calibrated_sample()
    approve, decline = select_thresholds(y, p, COSTS)
    assert decline == pytest.approx(breakeven_pd(COSTS), abs=0.025)
    assert approve == pytest.approx(decline * COSTS.refer_ratio)
    assert approve < decline


def test_selected_threshold_beats_extremes():
    y, p = calibrated_sample()
    _, decline = select_thresholds(y, p, COSTS)
    assert total_cost(y, p, decline, COSTS) <= total_cost(y, p, 0.6, COSTS)
    assert total_cost(y, p, decline, COSTS) <= total_cost(y, p, 0.005, COSTS)


def test_config_override_and_validation():
    y, p = calibrated_sample(5000)
    assert select_thresholds(y, p, COSTS, ThresholdsCfg(approve_below=0.02, decline_at_or_above=0.2)) == (0.02, 0.2)
    with pytest.raises(ValueError):
        select_thresholds(y, p, COSTS, ThresholdsCfg(approve_below=0.5, decline_at_or_above=0.1))


def test_decide_and_bands():
    assert decide(0.01, 0.04, 0.09) == APPROVE
    assert decide(0.05, 0.04, 0.09) == REFER
    assert decide(0.09, 0.04, 0.09) == DECLINE
    cfg = RiskBandsCfg()
    assert [risk_band(v, cfg) for v in (0.01, 0.05, 0.2)] == ["Low", "Medium", "High"]


def test_ks_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert ks_statistic(y, np.r_[np.zeros(50), np.ones(50)]) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert ks_statistic(rng.integers(0, 2, 5000), rng.random(5000)) < 0.06


def test_platt_calibration_restores_base_rate():
    rng = np.random.default_rng(2)
    p_true = rng.beta(1, 12, 50000)
    y = (rng.random(50000) < p_true).astype(int)
    inflated = 1 / (1 + np.exp(-(np.log(p_true / (1 - p_true)) + 2.0)))  # as if trained with class weights
    cal = PlattCalibrator().fit(inflated, y).predict(inflated)
    assert abs(cal.mean() - y.mean()) < 0.003
    assert calibration_table(y, cal)["abs_gap"].max() < 0.02


def test_psi_identical_is_small_and_shift_is_flagged():
    rng = np.random.default_rng(3)
    base = rng.normal(size=20000)
    assert psi_samples(base, rng.normal(size=20000)) < 0.1
    assert 0.1 < psi_samples(base, rng.normal(0.4, 1, 20000)) < 0.25  # moderate
    assert psi_samples(base, rng.normal(1.0, 1, 20000)) > 0.25   # major


def test_psi_status_thresholds():
    cfg = DriftCfg()
    assert status_for(0.05, cfg).startswith("OK")
    assert status_for(0.15, cfg).startswith("WARN")
    assert status_for(0.30, cfg).startswith("ALERT")
