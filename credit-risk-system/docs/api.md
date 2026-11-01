# API reference

Start: `python -m uvicorn creditrisk.api:app` (needs a trained model; interactive docs at `/docs`).
No authentication: demo only. Responses below are real outputs from the synthetic-trained model unless marked otherwise.

## GET /health
```json
{"status":"ok","model_version":"20261101-100143","api_version":"0.1.0"}
```
`status` is `"degraded"` with `model_version: null` if no model has been trained.

## POST /assess
Request (`age`, `monthly_income`, `dependents` are optional/nullable; unknown fields are rejected):
```json
{"application_id":"APP-001","revolving_utilization":0.40,"age":38,"past_due_30_59":0,"debt_ratio":0.35,
 "monthly_income":4500,"open_credit_lines":7,"past_due_90":0,"real_estate_loans":1,"past_due_60_89":0,"dependents":2}
```
Response 200:
```json
{"application_id":"APP-001","probability_of_default":0.067348,"credit_score":563,"risk_band":"Medium","decision":"REFER",
 "reasons":["Revolving credit utilization is moderate (40%)","Monthly income is moderate ($4,500)"],
 "model_version":"20261101-100143"}
```
| Status | When |
|---|---|
| 200 | Assessed and stored |
| 409 | `application_id` already exists (nothing is overwritten) |
| 422 | Invalid input: negative values, wrong types, missing required field, bad id (allowed: letters, digits, `_`, `-`, max 64), unknown field |
| 503 | No active model |

Validation limits: `age` 18-110; past-due counts 0-98; `open_credit_lines`/`real_estate_loans` 0-100; `dependents` 0-30; `monthly_income` >= 0.
`age` is stored but never used for scoring.

## POST /applications/{id}/outcome
Record the real repayment result (feedback for retraining).
```json
{"defaulted": false}
```
201: `{"application_id":"APP-001","defaulted":false,"recorded":true}`. 404 unknown id. 409 outcome already recorded. 422 invalid body.

## GET /referral-queue
REFER decisions with no outcome yet, oldest first.
```json
{"count":1,"items":[{"application_id":"APP-001","decided_at":"2026-11-01T09:58:21+00:00","pd":0.0673,"credit_score":563,
 "risk_band":"Medium","decision":"REFER","reasons":["..."],"model_version":"20261101-095717"}]}
```
(Captured from the earlier smoke test model `20261101-095717`, which was later deleted and retrained identically.)

## GET /applications/{id}
Stored decision plus the applicant payload and outcome (`null` until recorded). 404 if unknown.

## Field meanings
`probability_of_default`: calibrated PD within 2 years. `credit_score`: 600 = 50:1 good:bad odds, +20 points doubles the odds.
`risk_band`: Low < 3%, Medium 3-10%, High >= 10% (configurable). `decision` thresholds are in the model artifact (see model card).
