# Customer Churn Prediction — Retail Banking

A complete, end-to-end machine learning project for predicting customer churn in a European retail bank. Built as part of a Data Science internship, the project covers exploratory data analysis, feature engineering, multi-model comparison, threshold tuning, SHAP explainability, and an interactive Streamlit dashboard.

**Research paper:** [ResearchGate — DOI: 10.13140/RG.2.2.14869.08163](https://www.researchgate.net/publication/409276262_Predicting_Customer_Churn_in_Retail_Banking)

---

## The Problem

Retail banks lose a meaningful share of their customer base to attrition each year. Reacting after a customer signals they want to leave is expensive and often too late. This project builds a system that assigns every customer a churn probability score *before* they leave, and explains in plain terms which factors drove that score — giving relationship managers enough lead time to act.

---

## Dataset

- 10,000 customer records from a European retail bank (France, Germany, Spain)
- 14 original features: demographic, account behaviour, and engagement attributes
- Target variable: `Exited` (binary — 1 = churned, 0 = retained)
- Class imbalance: ~20.4% churn rate

| Feature | Description |
|---|---|
| CreditScore | Customer credit score (350–850) |
| Geography | Country: France, Germany, Spain |
| Gender | Male / Female |
| Age | Age in years (18–92) |
| Tenure | Years as a bank customer (0–10) |
| Balance | Account balance (€) |
| NumOfProducts | Number of bank products held (1–4) |
| HasCrCard | Holds a credit card (binary) |
| IsActiveMember | Active customer (binary) |
| EstimatedSalary | Estimated annual salary (€) |
| Exited | **Target** — whether the customer churned |

---

## Key Findings

- **NumOfProducts is the dominant predictor.** Customers with 2 products churn at just 7.6%. Those with 3 products churn at 82.7%. Those with 4 products: 100% — across a small but consistent sample.
- **Germany churns at twice the rate** of France and Spain (32.4% vs ~16%). Root cause is unknown from data alone and warrants a market-specific review.
- **Inactive members churn at nearly double** the rate of active members (26.9% vs 14.3%).
- **Age is the strongest numeric predictor** (r = 0.29). Churned customers cluster in the 40–55 age range.
- **Credit score, salary, and tenure are not predictive** — risk is driven by engagement and product fit, not financial profile.
- **Counter-intuitive finding:** Zero-balance customers churn *less* (13.8%) than customers with a balance (24.1%).

---

## Methodology

### Feature Engineering
Four engineered features were added on top of the raw columns:

| Feature | Formula | Rationale |
|---|---|---|
| `Balance_to_Salary` | Balance / (Salary + 1), capped at 99th percentile | Relative financial standing; raw ratio was unstable due to near-zero salaries |
| `Products_per_Year` | NumOfProducts / (Tenure + 1) | Rate of product accumulation |
| `Age_Tenure_Interaction` | Age × Tenure | Interaction between age and relationship length |
| `Engagement_Score` | IsActiveMember × NumOfProducts | Whether engagement and product count compound |

### Models Evaluated

Six models were benchmarked, in order of structural complexity:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.58 | 0.19 | 0.28 | 0.773 |
| Logistic Regression (balanced) | 0.38 | 0.71 | 0.50 | 0.775 |
| Decision Tree | 0.44 | 0.77 | 0.56 | 0.825 |
| Random Forest | 0.55 | 0.71 | 0.62 | 0.866 |
| **Gradient Boosting** | **0.76** | **0.48** | **0.59** | **0.867** |
| XGBoost | 0.52 | 0.73 | 0.61 | 0.859 |

*All metrics above are at the default 0.5 threshold.*

### Threshold Tuning

The default 0.5 threshold was found to be poorly calibrated for this imbalanced problem — Gradient Boosting achieved only 48% recall at that cutoff despite having the highest AUC. A threshold sweep across all models at fixed recall targets revealed that Gradient Boosting achieved the **best precision at every recall level** once properly tuned.

Final threshold chosen: **0.206**, based on the business reasoning that missing a churner (lost customer lifetime value) costs more than a false alarm (one unnecessary retention contact).

| At threshold 0.206 | Value |
|---|---|
| Recall (churn class) | 0.75 |
| Precision (churn class) | 0.52 |
| F1-score | 0.62 |
| ROC-AUC | 0.867 |
| Churners correctly flagged | 306 / 407 |
| Reduction in missed churners vs. baseline | ~70% |

### Cross-Validation

5-fold stratified cross-validation confirmed the results are stable:

| Metric | Mean | Std Dev |
|---|---|---|
| ROC-AUC | 0.865 | ±0.007 |
| Recall | 0.479 | ±0.030 |
| Precision | 0.763 | ±0.014 |

The low standard deviation in AUC (±0.007) across folds confirms the model's discriminative ability is not an artefact of a single favourable split.

### Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was applied using `TreeExplainer` to explain both global feature importance and individual customer predictions.

Top drivers (globally, across all customers):
1. `NumOfProducts` — dominant, non-linear effect
2. `Age` — moderate positive effect (older → higher risk)
3. `Geography_Germany` — clear positive contribution
4. `Gender` — females show higher churn contribution
5. `IsActiveMember` — inactive → higher risk

`Tenure`, `HasCrCard`, and `Geography_Spain` were consistently negligible across all three analyses (EDA, feature importance, and SHAP).

---

## Repository Structure

```
customer-churn-prediction/
│
├── data/
│   ├── European_Bank.csv           # Raw dataset
│   └── df_model_cleaned.csv        # Cleaned and engineered features
│
├── model/
│   ├── final_model_gradient_boosting.pkl   # Trained Gradient Boosting model
│   └── model_config.txt                    # Decision threshold and feature list
│
├── app.py                          # Streamlit dashboard (4 modules)
├── config.toml                     # Streamlit theme configuration
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment
└── Retail_Banking_Churn_Prediction.pptx   # Project presentation
```

---

## Dashboard

The Streamlit dashboard has four modules:

| Module | What it does |
|---|---|
| **① Risk Calculator** | Input a customer's profile and get an instant churn probability with SHAP waterfall explanation |
| **② Probability Distribution** | View predicted probability distributions, confusion matrix, ROC curve, and a live threshold explorer |
| **③ Feature Importance** | EDA charts, model importance rankings, and SHAP summary plots |
| **④ What-If Simulator** | Adjust a customer's engagement and product values and watch the risk score respond in real time |

### Running locally

```bash
# Clone the repository
git clone https://github.com/Himanshu-Rai06/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

Python 3.11 recommended (see `runtime.txt`).

---

## Research Paper

The full methodology, statistical analysis, and results are documented in the accompanying research paper:

> Rai, H. (2026). *Predicting Customer Churn in Retail Banking: A Machine Learning Approach to Identifying At-Risk Customers.*
> DOI: [10.13140/RG.2.2.14869.08163](https://www.researchgate.net/publication/409276262_Predicting_Customer_Churn_in_Retail_Banking)

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data analysis | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Modelling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Model serialisation | joblib |

---

## Author

**Himanshu Rai**
Data Science Intern
[GitHub](https://github.com/Himanshu-Rai06) · [ResearchGate](https://www.researchgate.net/publication/409276262_Predicting_Customer_Churn_in_Retail_Banking)
