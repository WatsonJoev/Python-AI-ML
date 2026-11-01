import math

import numpy as np
import pandas as pd
import pytest

from creditrisk.config import ScoreCfg, ScorecardCfg
from creditrisk.data import CleaningParams
from creditrisk.features import prepare
from creditrisk.scorecard import Scorecard, WoeBinner, make_edges, pd_to_score, score_to_pd, woe_table

SCORE = ScoreCfg()  # base 600 @ 50:1, pdo 20


def test_score_scaling_base_point():
    assert pd_to_score(1 / 51, SCORE) == pytest.approx(600.0)  # odds good:bad = 50:1


def test_score_doubles_odds_adds_pdo():
    # 50:1 -> 100:1 odds should be +20 points
    assert pd_to_score(1 / 101, SCORE) - pd_to_score(1 / 51, SCORE) == pytest.approx(20.0)


def test_score_monotone_and_invertible():
    pds = np.array([0.01, 0.05, 0.2, 0.6])
    scores = pd_to_score(pds, SCORE)
    assert np.all(np.diff(scores) < 0)
    assert score_to_pd(scores, SCORE) == pytest.approx(pds)


def test_woe_sign_and_iv():
    x = np.array([0] * 500 + [1] * 500)
    y = np.array([0] * 475 + [1] * 25 + [0] * 400 + [1] * 100)
    t = woe_table(x, y, edges=np.array([0.0]))
    assert t.iloc[0]["woe"] > 0 > t.iloc[1]["woe"]  # safer bin has the higher WoE
    pct_good, pct_bad = t["good"] / 875, t["bad"] / 125
    assert t["iv"].sum() == pytest.approx(((pct_good - pct_bad) * np.log(pct_good / pct_bad)).sum(), rel=0.05)
    assert t["iv"].sum() > 0.3


def test_informative_feature_has_higher_iv_than_noise():
    rng = np.random.default_rng(0)
    n = 20000
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(signal * 1.5 - 3)))).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    iv = WoeBinner(ScorecardCfg()).fit(X, pd.Series(y)).iv_table().set_index("feature")["iv"]
    assert iv["signal"] > 0.3 and iv["noise"] < 0.02


def test_make_edges_merges_small_bins():
    x = np.array([0] * 990 + [1] * 5 + [2] * 5)
    edges = make_edges(x, 10, 0.05)
    shares = np.bincount(np.searchsorted(edges, x, side="left")) / len(x)
    assert shares.min() >= 0.05 or len(edges) == 0


def test_scorecard_points_sum_to_score_and_match_scaling(artifact, env):
    sc: Scorecard = artifact["scorecard"]
    X = prepare(env["splits"].test.head(50), CleaningParams(**artifact["cleaning_params"]), env["settings"])
    direct = sc.score(X)
    assert direct == pytest.approx(pd_to_score(sc.predict_proba_raw(X), SCORE), abs=1e-6)
    bd = sc.breakdown(X.iloc[[0]])
    assert bd["points"].iloc[-1] == pytest.approx(direct[0])
    assert bd["points"].iloc[:-1].sum() == pytest.approx(direct[0])


def test_pd_to_score_handles_extremes():
    assert math.isfinite(float(pd_to_score(0.0, SCORE))) and math.isfinite(float(pd_to_score(1.0, SCORE)))
