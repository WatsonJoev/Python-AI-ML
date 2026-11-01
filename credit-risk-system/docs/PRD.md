# PRD: CreditRisk

Status: built as a portfolio project. Numbers referenced here are from synthetic data (see [model_card.md](model_card.md)).

## Problem
A bank receives loan applications and must decide quickly, consistently and explainably whether to lend.
Manual underwriting of every application is slow and inconsistent; a black-box score cannot be defended to a regulator
or an applicant. The bank needs a calibrated risk estimate, a decision rule tied to money, and reasons in plain language.

## Users
| User | Need |
|---|---|
| Credit analyst / underwriter | A referral queue of borderline cases with the reasons the model is unsure |
| Loan officer / channel system | An API that returns decision, score and reasons in under a second |
| Risk / model governance | Model card, fairness check, drift monitoring, versioned models, audit trail |
| Applicant (indirectly) | Understandable reasons if declined (adverse-action notice) |
| Data scientist | Reproducible training, safe retraining from outcomes |

## Goals
1. Estimate PD (serious delinquency within 2 years) with good rank-ordering and calibrated probabilities.
2. Convert PD to a score (base 600, 50:1 odds, 20 points to double the odds).
3. Decide APPROVE / REFER / DECLINE from a configurable cost assumption.
4. Return top reasons; decline reasons come only from factors that raise risk.
5. Keep a stored audit trail: application, decision, reasons, model version, later outcome.
6. Detect input drift (PSI) and retrain safely without auto-replacing the live model.

## Non-goals
Streaming/real-time feature pipelines, deep learning, cloud deployment, authentication/user management,
loan pricing or limit setting, collections, a UI, regulatory capital (IFRS 9 / Basel) modelling,
reject inference, legal sign-off on fair-lending compliance.

## Requirements
| Priority | Requirement | Status |
|---|---|---|
| P0 | Load Kaggle data if present, else labelled synthetic fallback | Done |
| P0 | Documented cleaning rules and engineered features | Done |
| P0 | WoE scorecard baseline + gradient boosting, same test set | Done |
| P0 | ROC-AUC, Gini, KS, PR-AUC, Brier, calibration table | Done |
| P0 | `POST /assess` with validation (422), duplicate id (409), PD/score/band/decision/reasons/version | Done |
| P0 | SQLite persistence; application + decision in one transaction | Done |
| P0 | Age excluded from the model; age-group performance report | Done |
| P0 | Same feature code for training and serving, with a test | Done |
| P1 | Cost-based thresholds from config | Done |
| P1 | SHAP reasons; scorecard points breakdown | Done |
| P1 | Outcome feedback endpoint, referral queue, application lookup | Done |
| P1 | PSI monitoring with 0.1 / 0.25 thresholds | Done |
| P1 | Retrain script writing a new versioned model, never auto-promoting | Done |
| P2 | Dockerfile | Written, NOT tested |
| P2 | Monotonic WoE bins, reject inference, auth, UI | Not done |

## Success metrics
Targets for a real deployment (the synthetic run is only a pipeline check):
- Gini >= 0.60 on a real out-of-time test set; KS >= 0.35.
- Calibration: per-decile gap between predicted PD and observed default rate within ~1 percentage point.
- Expected cost lower than "approve everyone" and than "decline everyone" (synthetic run: cost 5.78M vs 8.43M for approve-all).
- Referral rate within analyst capacity (synthetic run: 33.8%, probably too high; tune `refer_ratio`).
- PSI < 0.1 on all features monthly; any > 0.25 triggers investigation.
- API p95 latency under 500 ms (not measured).

## Risks
| Risk | Mitigation |
|---|---|
| Synthetic results mistaken for real | Every doc states the data source; labelled in the model artifact |
| Proxy discrimination (age-correlated features) | Age not used; performance and approval rates by age group reported |
| Miscalibration after class re-weighting | Platt calibration on validation, calibration table reported |
| Cost assumptions are guesses | Configurable; thresholds recomputed on retrain |
| Selection bias (outcomes only for approved loans) | Documented limitation; reject inference out of scope |
| Unsafe model promotion | Retrain writes a new file only; promotion is an explicit command |
| Pickle files executing code | Only load files produced by this project; never from users |
| PII exposure | No auth/encryption in the demo; not for production |

## Open questions
- What are the real cost of default (loss given default x exposure) and lost profit per good customer?
- Which jurisdiction's fair-lending rules apply (adverse-action wording, prohibited attributes)?
- Should REFER cases be resolved by a separate reviewer decision, not just an eventual outcome?
- How often is performance monitored, and who owns the promote decision?
- Is an out-of-time validation set available, to replace random stratified splits?
