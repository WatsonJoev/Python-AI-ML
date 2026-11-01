"""Demo: score three sample applicants (strong, borderline, weak) with the active model.

  python -m creditrisk.demo
"""
from __future__ import annotations

import sys

from .config import get_settings, setup_logging
from .data import CleaningParams
from .features import api_payload_to_frame, prepare
from .scoring import NoActiveModelError, Scorer

SAMPLES: dict[str, dict] = {
    "Strong applicant": dict(revolving_utilization=0.05, age=45, past_due_30_59=0, debt_ratio=0.15,
                             monthly_income=9000, open_credit_lines=8, past_due_90=0, real_estate_loans=1,
                             past_due_60_89=0, dependents=1),
    "Borderline applicant": dict(revolving_utilization=0.40, age=38, past_due_30_59=0, debt_ratio=0.35,
                                 monthly_income=4500, open_credit_lines=7, past_due_90=0, real_estate_loans=1,
                                 past_due_60_89=0, dependents=2),
    "Weak applicant": dict(revolving_utilization=0.98, age=29, past_due_30_59=3, debt_ratio=0.90,
                           monthly_income=2200, open_credit_lines=12, past_due_90=2, real_estate_loans=0,
                           past_due_60_89=2, dependents=3),
}


def main() -> int:
    setup_logging()
    settings = get_settings()
    try:
        scorer = Scorer.from_active(settings)
    except NoActiveModelError as exc:
        print(exc)
        return 1
    art = scorer.artifact
    print(f"Model {scorer.version} | trained on: {art['data_source']}")
    print(f"Cut-offs: APPROVE if PD < {scorer.approve_below:.3f}, DECLINE if PD >= {scorer.decline_at:.3f}, else REFER\n")
    sc = art["scorecard"]
    params = CleaningParams(**art["cleaning_params"])
    for name, applicant in SAMPLES.items():
        r = scorer.assess(applicant)
        print(f"=== {name} ===")
        print(f"  Decision: {r.decision:<8} PD: {r.pd:6.2%}   Score: {r.credit_score}   Risk band: {r.risk_band}")
        print("  Reasons:" if r.decision != "APPROVE" else "  Strongest favourable factors:")
        for reason in r.reasons:
            print(f"    - {reason}")
        feats = prepare(api_payload_to_frame(applicant), params, settings)
        card = sc.breakdown(feats)
        print(f"  Baseline scorecard score: {card.iloc[-1]['points']:.0f} (points breakdown below)")
        print(card.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
