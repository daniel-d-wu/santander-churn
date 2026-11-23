# Santander customer churn: discrete-time hazard model

Monthly churn (account closure) model for a retail-bank customer panel, built as a **discrete-time survival model**: a logistic regression on customer-month (person-period) data with a tenure-based baseline hazard, lagged behavioural features, and out-of-time validation. A LightGBM model on the same table serves as the challenger.

Data: [Kaggle — Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation/data) (`train_ver2.csv`, monthly snapshots Jan-2015 to May-2016). The data is **not included** in this repo; see *Reproducing* below.

## Approach

| Step | What | Key decisions |
|---|---|---|
| 1. Data | 10% customer sample (`ncodpers % 10 == 0`), all 17 months, compact dtypes | Sample customers, not rows, so each panel stays complete; `(ncodpers, fecha_dato)` asserted unique |
| 2. Target | `churn = 1` on a customer's last month in the panel before May-2016; May-2016 censored | Validated against the bank's closure flag (`indrel = 99`) and zero-product holdings on the final row |
| 3. Clean | Normalise codes, cap age, flag missing income | Jan–Jun 2015 exits come entirely from records with no attributes, so outcomes are modelled from Jul-2015 only |
| 4. Person-period table | One row per customer-month at risk, Jul-2015 to Apr-2016 | Time scale = **tenure** (months since `fecha_alta`), with delayed entry for customers already on book |
| 5. Features | Product holdings, activity, 3-month product trend, income-missing flag, age, segment, channel, December | Dynamic features **lagged one month**: the same-month closure flag catches 95% of churners, i.e. it is the outcome itself |
| 6. Split + feature analysis | Dev = Jul–Dec 2015, out-of-time (OOT) = Jan–Apr 2016 | IV / WoE, monotonicity, PSI, redundancy and tenure-interaction checks on dev only |
| 7. Model | Logit with natural cubic spline on log(tenure + 1) as the baseline | Baseline chosen by customer-grouped CV over constant, linear, polynomial, tenure bands and splines; feature × new-customer interactions |
| 8–9. Evaluation, prune + L2 | One-time OOT evaluation; prune non-significant terms; ridge penalty via 1-SE rule on dev CV | Improves decile calibration ~40% with no loss in ranking |
| 10. Challenger | LightGBM on the same rows and split | Measures what the hand-specified logit leaves on the table |

## Results (OOT, Jan–Apr 2016, 750 churners in the 10% sample)

| Model | AUC | Log loss ×1000 | Top-10% capture | Top-5% capture | Avg decile calibration error |
|---|---|---|---|---|---|
| Tenure-only baseline | 0.731 | 13.105 | 48.5% | 42.4% | 0.058 pp |
| Logit, pruned + L2 | 0.800 | 12.471 | 57.7% | 48.4% | **0.033 pp** |
| LightGBM | **0.817** | **12.221** | **60.0%** | **54.0%** | 0.067 pp |

The logit captures ~80% of LightGBM's AUC gain over the tenure-only baseline and is better calibrated; LightGBM's edge is concentrated in the riskiest ~5% of customer-months.

## Repo contents

| File | Contents |
|---|---|
| `santander_churn.ipynb` | Full analysis, steps 1–10, with outputs and plots |
| `docs/santander_data_dictionary.xlsx` | All 48 raw fields: Spanish → English, definitions, observed values, cleanup notes |

## Reproducing

1. Accept the competition rules on Kaggle and place your API token at `~/.kaggle/kaggle.json`.
2. Run the notebook from the top: it downloads `train_ver2.csv.zip` into `data/`, builds the 10% sample, and caches it as `data/train_sample10.parquet`.
3. Python packages: `pandas`, `numpy`, `pyarrow`, `statsmodels`, `patsy`, `scikit-learn`, `lightgbm`, `matplotlib`, `kaggle`.
