# Architecture

## Components
| Module (`src/creditrisk/`) | Responsibility |
|---|---|
| `config.py` | Typed settings from `config.yaml` (pydantic); paths resolved relative to the config file; `CREDITRISK_CONFIG` override |
| `data.py` | Kaggle loader, synthetic generator, `clean()` rules, `CleaningParams` (train-only statistics), stratified split |
| `features.py` | `engineer()` and `prepare()`: the single feature pipeline used by training and serving; API-field to column mapping |
| `scorecard.py` | Quantile binning, WoE/IV, logistic regression, points per bin, score scaling (`pd_to_score`) |
| `train.py` | Fits scorecard + gradient boosting, calibrates, chooses thresholds, writes the versioned artifact and `reports/` |
| `evaluate.py` | Metrics (ROC-AUC, Gini, KS, PR-AUC, Brier), calibration table, Platt calibrator, cost-based thresholds, risk bands, age-group report |
| `explain.py` | SHAP `TreeExplainer` and the feature-to-sentence templates |
| `scoring.py` | Model registry (versioned files + `ACTIVE_MODEL` pointer) and `Scorer` |
| `store.py` | SQLite access: transactions, referral queue, outcomes, feedback export |
| `api.py` | FastAPI app factory, request/response models |
| `drift.py` | PSI, reference profile, report, CLI |
| `retrain.py` | Feedback merge, new model, comparison, explicit `--promote` |
| `demo.py` | Scores three sample applicants |

## Data flow
**Training** (`python -m creditrisk.train`)
1. `load_raw` uses `data/cs-training.csv` if it exists, else generates a synthetic frame (and says so in the log and the artifact).
2. Stratified split: 60% train / 20% validation / 20% test, `seed=42`.
3. `fit_cleaning_params(train)` (median income, median age) -> `prepare()` on every split.
4. Scorecard: bins + WoE + IV on train; features with IV < 0.02 dropped; logistic regression on WoE.
5. Gradient boosting on train with balanced sample weights.
6. Platt calibrators (one per model) fitted on **validation**; thresholds chosen on **validation**.
7. Test set evaluation of both models, reports written, artifact saved as `models/model_<UTC timestamp>.joblib`.
   If no `ACTIVE_MODEL` exists the new version becomes active; otherwise it is left alone.

**Serving** (`POST /assess`)
1. Pydantic validates the body (422 on failure, unknown fields rejected).
2. Duplicate check (409), then `Scorer.assess`: `prepare()` -> gradient boosting probability -> Platt calibration -> PD.
3. Score = `pd_to_score(PD)`; band from config PD cut-offs; decision from the artifact's thresholds.
4. SHAP values of the same feature row -> reasons.
5. `save_assessment` inserts application + decision in a single transaction.

## Training vs serving
| | Training | Serving |
|---|---|---|
| Input | DataFrame with Kaggle column names | API dict -> one-row DataFrame (`api_payload_to_frame`) |
| Cleaning | `clean(df, params)` | identical call, `params` loaded from the artifact |
| Features | `prepare()` | identical `prepare()` |
| Parity | `tests/test_parity_explain.py`: row-by-row features equal batch features; served PD equals offline PD | |

Everything needed to score is inside one artifact: models, calibrators, cleaning parameters, thresholds,
data-source label, test metrics and the PSI reference profile. This prevents mixing a model with the wrong preprocessing.

## Retraining
`retrain.py` reads recorded outcomes, rebuilds their features from the stored applicant payloads, appends them to the
**training** split only, retrains with a new version id, and evaluates the active and candidate models on the same
original test split. The candidate is saved as a new file; `ACTIVE_MODEL` is unchanged. Promotion is
`python -m creditrisk.retrain --promote <version>`. A running API process caches its scorer, so restart it after promotion.

## Storage schema (SQLite, `data/creditrisk.db`)
```sql
applications(application_id TEXT PK, created_at TEXT, payload_json TEXT)
decisions(application_id TEXT PK -> applications, decided_at TEXT, pd REAL, credit_score INTEGER,
          risk_band TEXT, decision TEXT CHECK IN ('APPROVE','REFER','DECLINE'), reasons_json TEXT, model_version TEXT)
outcomes(application_id TEXT PK -> applications, recorded_at TEXT, defaulted INTEGER CHECK IN (0,1))
```
Foreign keys are enforced (`PRAGMA foreign_keys=ON`). One connection per operation (safe with FastAPI's thread pool).
The referral queue is `decision='REFER'` with no outcome row yet.

## Files on disk
`models/model_<version>.joblib` (never overwritten), `models/ACTIVE_MODEL` (text pointer), `reports/*` (CSV, PNG, JSON, console logs).
