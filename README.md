# Skincare Behavior Intelligence — TPB/SDT Decision Cockpit

A production-ready Streamlit dashboard to analyze **Mumbai skincare survey** data, quantify **Theory of Planned Behaviour (TPB)** and **Self-Determination Theory (SDT)** levers, diagnose **Barriers/Motivators**, and simulate **ROI** scenarios for conversion and spend.

---

## 1) What this app does

- Ingests `skincare_survey_mumbai_120.csv` (or your upload) and performs:
  - **Overview**: audit, missingness, segment & gender mix, income/education, top localities.
  - **Segments & Personas**: KPIs by `User_Type`, chi-square crosstabs, compact persona cards.
  - **TPB/SDT**: builds composites (**ATT, SN, PBC, BI, AUT, COMP, REL**), computes **Cronbach’s α**, shows group differences (ANOVA/Kruskal) and **Spearman** correlations.
  - **Barriers & Motivators**: frequency, **co-occurrence network**, and “needle movers” for **BI**.
  - **Predict & Explain**:
    - **Classification**: `is_frequent = (User_Type == "Frequent")` via `LogisticRegression` with `ColumnTransformer` (OHE) + `StandardScaler`.
    - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC; **5-fold CV** (fallback to 3-fold if small).
    - Explainability: **Permutation importance**, **Partial Dependence** (top composites).
    - **Regression**: `RandomForestRegressor` on `Monthly_Spend` (users only) with RMSE/MAE/R² + CV.
  - **Recommendations & ROI**: sliders to shift composites (−1.0 to +1.0), conceptually reduce barriers, and re-score **Frequent share**, **Recommendation rate (proxy)**, and **Avg Spend**. Export **scored CSV**.

All charts use **matplotlib** (no seaborn). App is defensive: **column guards**, friendly warnings, and degradation when optional fields are missing.

---

## 2) Data dictionary (key fields)

- Demographics: `Name, Age, Gender, Locality, Occupation, Monthly_Income, Education`
- Behavior & outcomes: `User_Type, Daily_Skincare_Use, Monthly_Spend, Is_Satisfied, Recommendation, Future_Purchase_Intent, Switching_Brands`
- Marketing: `Info_Source, Brand_Preference, Purchase_Motivation, Barriers, Motivators`
- Likert items (1–5):  
  - **ATT1–ATT7** (*Attitude*)  
  - **SN1–SN6** (*Subjective Norms*)  
  - **PBC1–PBC5** (*Perceived Behavioral Control*)  
  - **BI1–BI4** (*Behavioral Intention*)  
  - **AUT1–AUT6** (*Autonomy*, with **AUT3** “pressure” item; reverse-coding toggle for α)  
  - **COMP1–COMP5** (*Competence*)  
  - **REL1–REL4** (*Relatedness*)

**Composites**: mean across available items per construct. If some items are missing, the app computes from the subset and displays a warning.

---

## 3) How to run

### Local
```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
