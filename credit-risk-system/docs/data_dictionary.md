# Data dictionary

**Datasets.** Primary: Kaggle "Give Me Some Credit" `cs-training.csv` (manual download into `data/`; ~150,000 rows, ~6.7% default).
Fallback: a synthetic frame with the same columns (60,000 rows, 7.03% defaults, ~20% missing income, ~2.6% missing dependents,
~1% utilization outliers, 96/98 special codes). **All results in this repository use the synthetic fallback.**

## Raw columns
| Column | API field | Meaning | Notes |
|---|---|---|---|
| `SeriousDlqin2yrs` | (target) | 1 = 90+ days past due within 2 years | ~7% positive |
| `RevolvingUtilizationOfUnsecuredLines` | `revolving_utilization` | Card/credit-line balance divided by limit | Ratio (0.35 = 35%); has huge outliers |
| `age` | `age` | Borrower age | **Not a model feature**; fairness reporting only |
| `NumberOfTime30-59DaysPastDueNotWorse` | `past_due_30_59` | Times 30-59 days late in 2 years | Kaggle uses 96/98 as special codes |
| `DebtRatio` | `debt_ratio` | Monthly debt payments / monthly income | Huge when income is missing (an amount, not a ratio) |
| `MonthlyIncome` | `monthly_income` | Monthly income | ~20% missing |
| `NumberOfOpenCreditLinesAndLoans` | `open_credit_lines` | Open loans and credit lines | |
| `NumberOfTimes90DaysLate` | `past_due_90` | Times 90+ days late | 96/98 codes |
| `NumberRealEstateLoansOrLines` | `real_estate_loans` | Mortgages and home-equity lines | |
| `NumberOfTime60-89DaysPastDueNotWorse` | `past_due_60_89` | Times 60-89 days late | 96/98 codes |
| `NumberOfDependents` | `dependents` | Family dependents | ~2.6% missing |

## Cleaning rules (`data.clean`, parameters in `config.yaml`)
| # | Rule | Why |
|---|---|---|
| 1 | `income_was_missing` = 1 if `MonthlyIncome` is missing | Missingness itself can be informative |
| 2 | Missing `MonthlyIncome` -> **training-split median** (stored in the model artifact) | Simple, robust; learned on train only to avoid leakage |
| 3 | `MonthlyIncome` clipped to [0, 50,000] | Extreme incomes distort bins and trees |
| 4 | Missing `NumberOfDependents` -> 0; clipped to [0, 10] | Missing usually means none recorded |
| 5 | Missing `age` -> training median; clipped to [18, 100] | Only used in the fairness report |
| 6 | `RevolvingUtilization...` clipped to [0, 1.5] | Values above ~1.5 are data errors or extreme; 1.5 preserves "over limit" |
| 7 | `DebtRatio` clipped to [0, 5.0] | Values in the thousands are incomes mistaken for ratios |
| 8 | Three past-due counts clipped to [0, 10] | Neutralises the 96/98 special codes while keeping "very delinquent" |
| 9 | `NumberOfOpenCreditLinesAndLoans` clipped to [0, 30]; `NumberRealEstateLoansOrLines` to [0, 10] | Outliers |

Caveat: clipping 96/98 codes to 10 treats them as "very delinquent" rather than as a separate category.

## Engineered features (`features.engineer`)
| Feature | Definition |
|---|---|
| `TotalPastDue` | 30-59 + 60-89 + 90+ counts |
| `IncomePerDependent` | `MonthlyIncome / (NumberOfDependents + 1)` |
| `HighDebtRatioFlag` | 1 if `DebtRatio > 0.43` (common debt-to-income guideline; configurable) |
| `MissingIncomeFlag` | copy of `income_was_missing` |

## Model feature set (13)
`RevolvingUtilizationOfUnsecuredLines, DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberRealEstateLoansOrLines,
NumberOfDependents, NumberOfTime30-59DaysPastDueNotWorse, NumberOfTime60-89DaysPastDueNotWorse, NumberOfTimes90DaysLate,
TotalPastDue, IncomePerDependent, HighDebtRatioFlag, MissingIncomeFlag`

## Information Value from the synthetic run (`reports/iv_table.csv`)
| Feature | IV | Strength |
|---|---|---|
| RevolvingUtilization | 0.606 | "suspiciously strong" (a real-data red flag; here an artefact of the generator) |
| TotalPastDue | 0.395 | strong |
| 30-59 days late | 0.208 | medium |
| 90+ days late | 0.205 | medium |
| 60-89 days late | 0.129 | medium |
| DebtRatio | 0.091 | weak |
| MonthlyIncome | 0.078 | weak |
| HighDebtRatioFlag | 0.046 | weak |
| IncomePerDependent | 0.040 | weak |
| OpenCreditLines, RealEstate, Dependents, MissingIncomeFlag | < 0.003 | dropped from the scorecard (IV < 0.02) |

These values describe the synthetic generator, not real borrowers.
