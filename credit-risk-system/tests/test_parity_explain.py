import numpy as np
import pandas as pd
import pytest

from conftest import STRONG, WEAK
from creditrisk.data import CleaningParams
from creditrisk.evaluate import APPROVE, DECLINE
from creditrisk.explain import reasons_for_row
from creditrisk.features import MODEL_FEATURES, api_payload_to_frame, prepare
from creditrisk.scoring import Scorer


def test_age_is_not_a_model_feature():
    assert "age" not in MODEL_FEATURES


def test_train_serve_parity_batch_vs_single(env, artifact, settings):
    """Features built row-by-row (serving path) must equal features built in a batch (training path)."""
    params = CleaningParams(**artifact["cleaning_params"])
    raw = env["splits"].test.drop(columns=["SeriousDlqin2yrs"]).head(200)
    batch = prepare(raw, params, settings)
    single = pd.concat([prepare(raw.iloc[[i]], params, settings) for i in range(len(raw))])
    pd.testing.assert_frame_equal(batch, single)


def test_serving_pd_equals_training_pipeline_pd(artifact, settings):
    """Scorer.assess on API-style dicts gives the same PD as the offline training-time computation."""
    scorer = Scorer(settings, artifact)
    params = CleaningParams(**artifact["cleaning_params"])
    for payload in (STRONG, WEAK, {**STRONG, "monthly_income": None, "dependents": None}):
        offline = prepare(api_payload_to_frame(payload), params, settings)
        expected = artifact["gb_calibrator"].predict(artifact["gb"].predict_proba(offline)[:, 1])[0]
        assert scorer.assess(payload).pd == pytest.approx(expected)


def test_changing_age_does_not_change_the_result(settings, artifact):
    scorer = Scorer(settings, artifact)
    a, b = scorer.assess({**STRONG, "age": 22}), scorer.assess({**STRONG, "age": 80})
    assert a.pd == b.pd and a.reasons == b.reasons


def test_weak_scores_worse_than_strong(settings, artifact):
    scorer = Scorer(settings, artifact)
    s, w = scorer.assess(STRONG), scorer.assess(WEAK)
    assert w.pd > s.pd and w.credit_score < s.credit_score


def test_decline_reasons_only_use_risk_increasing_features():
    shap_row = np.zeros(len(MODEL_FEATURES))
    shap_row[0] = 0.8     # utilization raises risk
    shap_row[2] = -0.9    # income LOWERS risk -> must not appear as a decline reason
    row = pd.Series({f: 0.0 for f in MODEL_FEATURES})
    row[MODEL_FEATURES[0]] = 0.95
    row[MODEL_FEATURES[2]] = 9000
    assert reasons_for_row(shap_row, row, DECLINE) == ["Revolving credit utilization is very high (95%)"]


def test_approve_reasons_use_risk_reducing_features():
    shap_row = np.zeros(len(MODEL_FEATURES))
    shap_row[2] = -0.9
    row = pd.Series({f: 0.0 for f in MODEL_FEATURES})
    row[MODEL_FEATURES[2]] = 9000
    assert reasons_for_row(shap_row, row, APPROVE) == ["Monthly income is good ($9,000)"]


def test_weak_applicant_gets_real_reasons(settings, artifact):
    r = Scorer(settings, artifact).assess(WEAK)
    assert 1 <= len(r.reasons) <= settings.reasons.max_reasons
    assert any("utilization" in x.lower() or "late" in x.lower() for x in r.reasons)
