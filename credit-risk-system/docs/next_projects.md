# Next banking AI/ML projects using the same skeleton

Reusable pieces: `config.yaml` + typed settings, `data.py` (loader/cleaning/split), one shared `features.prepare` for train and serve,
`evaluate.py` (metrics, calibration, cost thresholds), `store.py` (SQLite audit trail), FastAPI app factory, PSI drift, versioned
model files with manual promotion, and the docs set (PRD, model card, API, interview guide). Datasets listed are suggestions; check licences before use.

| Project | Dataset idea | Model type | Difficulty | What changes from CreditRisk |
|---|---|---|---|---|
| Fraud detection (card transactions) | Kaggle "Credit Card Fraud Detection" (ULB, 0.17% fraud) or the synthetic PaySim | Gradient boosting / isolation forest as a baseline; PR-AUC, recall at fixed alert rate | Medium | Far stronger imbalance; threshold set by analyst alert capacity; time-based split is essential |
| Customer churn | Kaggle "Bank Customer Churn" or Telco churn | Logistic regression + gradient boosting; survival model as a stretch | Easy | Business metric is retention uplift; explanations drive offers |
| Loan amount / limit prediction | Lending Club accepted loans, or synthetic income/credit history | Regression (gradient boosting, quantile regression); MAE/MAPE plus affordability rules | Medium | Output is an amount; must combine with PD and an affordability cap |
| AML transaction monitoring | IBM AML synthetic transactions or PaySim | Rules + anomaly detection + supervised ranking of alerts; graph features as a stretch | Hard | Few labels, regulatory reporting, alert-queue prioritisation and investigator feedback loop |
| Customer segmentation | Synthetic transaction summaries or the UCI "Bank Marketing" data | K-means / Gaussian mixture / hierarchical clustering; silhouette, stability, business profiling | Easy | Unsupervised: no target, evaluation is stability and usefulness |
| Collections prioritisation | Lending Club delinquent loans, or CreditRisk's outcomes table | Propensity-to-pay model (boosting) + expected-recovery ranking; uplift modelling as a stretch | Medium | Rank accounts by expected recovered amount per agent-hour |
| KYC document extraction | Synthetic ID/utility-bill images, public invoice/receipt datasets (e.g. SROIE) | OCR (Tesseract/PaddleOCR) + field extraction with regex/layout rules; a small LLM or layout model as a stretch | Hard | Unstructured input; field-level accuracy, confidence thresholds, human verification queue; handle PII carefully |
| Banking FAQ chatbot | Public bank FAQ pages or a self-written FAQ set | Retrieval (TF-IDF or embeddings) + templated answers; LLM with retrieval as a stretch | Medium | Evaluate answer accuracy and refusal on out-of-scope or sensitive requests; never expose account data without auth |
