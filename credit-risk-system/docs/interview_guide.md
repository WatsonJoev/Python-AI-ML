# Interview guide

Be upfront in interviews: results are from **synthetic data**. Saying so is a strength, not a weakness.

## Resume bullets (pick 3-4)
- Built an end-to-end credit risk system (Python, scikit-learn, FastAPI, SQLite) that returns probability of default, a 600/50:1/PDO-20 credit score,
  an approve/refer/decline decision and SHAP-based adverse-action reasons for loan applicants.
- Compared a WoE/Information-Value logistic scorecard with gradient boosting on one held-out test set using Gini, KS, PR-AUC, Brier and calibration tables;
  handled 7% class imbalance with stratified splits, sample weights and Platt calibration.
- Derived approve/refer/decline cut-offs from a configurable default-vs-lost-profit cost model (cut expected cost 31% vs approving everyone on synthetic data);
  excluded age from the model and reported performance and approval rates by age group.
- Added PSI drift monitoring, an outcome-feedback API and a retraining script that writes versioned models and never auto-promotes; 48 pytest tests incl. train/serve parity.

## 2-minute explanation
"Banks need to decide on loans quickly, consistently and be able to explain the decision. I built a service where you send an applicant and get back a probability of
default, a credit score, a decision and the main reasons. I used the Give Me Some Credit schema; since I can't ship the Kaggle file, the repo falls back to a synthetic
dataset with the same columns and I label every result as synthetic. I cleaned the data, capped outliers, and trained two models: the traditional scorecard (logistic
regression on weight-of-evidence bins, with an IV table and points per bin) and gradient boosting. Defaults are rare, so accuracy is meaningless; I report Gini, KS, PR-AUC, Brier and a
calibration table, and I calibrate the boosting model because class weights distort probabilities. I turn PD into approve, refer or decline using a cost assumption: a default costs
ten times the profit from a good customer. Reasons come from SHAP and only features that raise risk appear as decline reasons. I left age out of the model because of fair-lending rules and
check results by age group. The API stores each application and decision in SQLite in one transaction, accepts repayment outcomes, and a retrain script produces a new versioned model that
a human must promote. On synthetic data both models reach about 0.77 AUC; I'd expect real data to differ, and I list that as a limitation."

## 15 likely questions
1. **Why not accuracy?** With 7% defaults, "approve everyone" is 93% accurate and useless. Use ranking (AUC/Gini/KS), precision-recall for the rare class, and Brier/calibration for probability quality.
2. **How did you handle class imbalance?** Stratified splits keep the rate equal; balanced sample weights in boosting; Platt calibration on validation restores true PDs; metrics that suit imbalance. I did not oversample (SMOTE) because it distorts PDs.
3. **What are Gini and KS?** Gini = 2*AUC - 1 (rank-ordering power). KS = max gap between the cumulative score distributions of bads and goods. Banks like both; Gini ~0.54 here (synthetic).
4. **What is WoE and IV?** WoE = ln(%good / %bad) in a bin; it linearises the effect on log-odds and handles missing/outliers. IV = sum((%good - %bad) * WoE) measures feature strength (<0.02 useless, >0.5 suspicious, often leakage).
5. **Why is a scorecard still used?** Transparent additive points, easy to audit and explain to regulators and customers, stable, simple to implement. Boosting may rank slightly better but is harder to govern.
6. **Why does boosting not beat the scorecard here?** The synthetic data were generated from a near-linear rule, so there is nothing non-linear to find. I would not claim it generalises either way.
7. **What is calibration and why does it matter?** Predicted 5% should mean about 5% default. Needed for pricing, expected loss and cost-based cut-offs. Checked with decile tables; fixed with Platt scaling on validation data.
8. **SHAP vs scorecard points?** Scorecard points are exact, global and additive by construction. SHAP attributes a non-linear model's output to features locally relative to a baseline; it is an approximation of behaviour, not causality, and interactions can give surprising attributions.
9. **How do you give decline reasons legally?** Adverse-action notices need specific principal reasons. I list only features with positive SHAP (raising risk) in plain English. A real deployment needs compliance-reviewed wording and mapping.
10. **What is data leakage and where could it occur?** Using information unavailable at decision time. I avoided it by fitting medians, bins and models on train only, calibrating/thresholding on validation, and touching the test set once. A very high IV (>0.5) or "too good" AUC should trigger a leakage hunt.
11. **What is PSI?** Population Stability Index compares a feature's current distribution with training across bins: sum((actual% - expected%) * ln(actual% / expected%)). <0.1 stable, 0.1-0.25 moderate, >0.25 major shift.
12. **Why exclude age?** Regulations (e.g. ECOA/Reg B in the US) restrict age in credit decisions; jurisdiction-specific. I still report approval rate and AUC by age group. I also note that excluding age made predicted risk flat across ages while actual risk differed.
13. **How are thresholds chosen?** Decline if expected loss of approving (PD x 10,000) exceeds expected profit lost by declining ((1-PD) x 1,000), i.e. PD above ~9%. I optimise total cost on validation data; a REFER band sits below. The costs are assumptions in config.
14. **How do you prevent train/serve skew?** One `prepare()` function and one artifact containing the cleaning statistics; a test compares row-by-row and batch features and offline vs served PD.
15. **How does retraining stay safe?** Outcomes are merged into the train split only, the candidate is compared with the active model on the same held-out test set, written as a new versioned file, and promoted manually. Also: outcomes only exist for approved loans (selection bias) and take about 2 years to mature.

Bonus: "What would you do next?" Out-of-time validation, monotonic binning, reject inference, bootstrap confidence intervals on Gini, auth + audit logging, a reviewer workflow for REFER cases.
