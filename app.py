# app.py
# Streamlit dashboard for skincare_survey_mumbai_120.csv
# Visual refresh: light Unilever-inspired theme (soft azure / mint / lilac) applied via CSS.
# Core functionality, charts, and data logic unchanged.

import io
import os
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx

import streamlit as st

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

warnings.filterwarnings("ignore")

# ------------------------------- #
# ---------- CONSTANTS ---------- #
# ------------------------------- #
RANDOM_STATE = 42

LIKERT_MAP = {
    "ATT": [f"ATT{i}" for i in range(1, 8)],
    "SN":  [f"SN{i}"  for i in range(1, 7)],
    "PBC": [f"PBC{i}" for i in range(1, 6)],
    "BI":  [f"BI{i}"  for i in range(1, 5)],
    "AUT": [f"AUT{i}" for i in range(1, 7)],
    "COMP":[f"COMP{i}" for i in range(1, 6)],
    "REL": [f"REL{i}" for i in range(1, 5)],
}

CORE_COLS = [
    "Name","Age","Gender","Locality","Occupation","Monthly_Income","Education",
    "User_Type","Daily_Skincare_Use","Product_Types_Used","Brand_Preference","Info_Source",
    "Purchase_Motivation","Monthly_Spend","Adverse_Effects","Is_Satisfied","Switching_Brands",
    "Awareness_of_Cosmetovigilance","Social_Media_Impact","Recommendation","Future_Purchase_Intent",
    "Barriers","Motivators"
]

# -------------------------------- #
# ---------- THEME CSS ----------- #
# -------------------------------- #
def apply_brand_theme():
    """
    Inject a light Unilever-inspired theme without altering charts or layouts.
    Palette:
      Primary:   #1F70C1 (Unilever Blue) used sparingly for accents
      Azure-50:  #F5FAFF (app bg)
      Azure-100: #E8F2FF (cards)
      Mint-100:  #EAFBF4 (soft success)
      Lilac-100: #F3F0FF (secondary accents)
      Slate-700: #2F3B52 (text)
    """
    st.markdown(
        """
        <style>
        :root{
          --pri:#1F70C1;
          --bg:#F5FAFF;
          --card:#E8F2FF;
          --card2:#F3F0FF;
          --mint:#EAFBF4;
          --text:#2F3B52;
          --muted:#6B7A90;
          --border:#D6E6FF;
          --shadow: 0 8px 22px rgba(31,112,193,0.10);
          --radius:18px;
        }
        /* App background */
        .stApp {
          background: linear-gradient(180deg, rgba(31,112,193,0.06) 0%, rgba(31,112,193,0.00) 40%), var(--bg);
        }
        /* Main container width + padding */
        .block-container{
          padding-top: 1.2rem;
          padding-bottom: 2rem;
        }
        /* Titles */
        h1, h2, h3, h4 {
          color: var(--text) !important;
          letter-spacing: 0.2px;
        }
        /* Subtle pill caption under H1 */
        .stMarkdown > p, .stCaption, .st-emotion-cache-17ziqus p {
          color: var(--muted) !important;
        }
        /* Cards: dataframe wrapper, metrics, info/warning boxes */
        .stDataFrame, .stTable, .stAlert, .stSuccess, .stInfo, .stWarning, .stError{
          border-radius: var(--radius) !important;
          box-shadow: var(--shadow);
        }
        /* Dataframe toolbar/background */
        div[data-testid="stDataFrame"] > div{
          background: var(--card) !important;
          border-radius: var(--radius) !important;
          border: 1px solid var(--border);
        }
        /* Metric cards */
        div[data-testid="stMetric"]{
          background: var(--mint);
          padding: 14px 16px;
          border-radius: var(--radius);
          border: 1px solid #DDF3EA;
          box-shadow: var(--shadow);
        }
        div[data-testid="stMetric"] label{
          color: var(--muted);
        }
        div[data-testid="stMetricValue"]{
          color: var(--text);
        }
        /* Sidebar */
        section[data-testid="stSidebar"]{
          background: linear-gradient(180deg, rgba(31,112,193,0.07) 0%, rgba(31,112,193,0.00) 60%), #FFFFFF;
          border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label{
          color: var(--text);
        }
        /* Buttons */
        .stButton>button{
          background: linear-gradient(180deg, #ffffff 0%, #E8F2FF 100%);
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 0.5rem 1rem;
          box-shadow: var(--shadow);
        }
        .stButton>button:hover{
          border-color: var(--pri);
          box-shadow: 0 10px 26px rgba(31,112,193,0.18);
        }
        /* Download buttons */
        .stDownloadButton>button{
          background: linear-gradient(180deg, #ffffff 0%, #E8F2FF 100%);
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 0.5rem 1rem;
          box-shadow: var(--shadow);
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"]{
          gap: 6px;
        }
        .stTabs [data-baseweb="tab"]{
          background: #FFFFFF;
          border: 1px solid var(--border);
          border-bottom-color: transparent;
          border-top-left-radius: 12px;
          border-top-right-radius: 12px;
          padding: 10px 14px;
          color: var(--muted);
        }
        .stTabs [aria-selected="true"]{
          background: var(--card2) !important;
          color: var(--text) !important;
          border-color: var(--pri) !important;
        }
        /* Selects, sliders */
        div[data-baseweb="select"]>div{
          background: #FFFFFF !important;
          border-radius: 10px;
          border: 1px solid var(--border);
        }
        .stSlider > div > div > div{
          background: linear-gradient(90deg, var(--pri), #77C3FF);
        }
        /* Charts canvases retain default styles; wrap in soft card */
        .stPlotlyChart, .stPyplot{
          background: #FFFFFF;
          border-radius: var(--radius);
          padding: 6px 6px 2px 6px;
          box-shadow: var(--shadow);
          border: 1px solid var(--border);
        }
        /* Info boxes tint */
        .stInfo{
          background: #F0F6FF !important;
          border: 1px solid var(--border) !important;
        }
        /* Small code blocks / captions */
        code, .stCode{
          background: #F7FAFF !important;
          color: var(--text) !important;
          border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------- #
# ---------- UTILITIES ----------- #
# -------------------------------- #
def set_page():
    st.set_page_config(
        page_title="Skincare Behavior Intelligence",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_brand_theme()  # <-- inject the theme early
    st.title("💧 Skincare Behavior Intelligence — TPB/SDT Decision Cockpit")
    st.caption("Manager-ready dashboard • Mumbai cohort (ages 21–30) • Likert composites & predictive levers")

def require_cols(df: pd.DataFrame, required: List[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file: io.BytesIO = None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    for try_path in ["./data/skincare_survey_mumbai_120.csv", "skincare_survey_mumbai_120.csv"]:
        if os.path.exists(try_path):
            return pd.read_csv(try_path)
    if "sample_df" in st.session_state and isinstance(st.session_state.sample_df, pd.DataFrame):
        return st.session_state.sample_df.copy()
    cols = CORE_COLS + sum(LIKERT_MAP.values(), [])
    return pd.DataFrame(columns=cols)

def strip_whitespace_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round().astype("Int64")
    if "Monthly_Spend" in df.columns:
        df["Monthly_Spend"] = pd.to_numeric(df["Monthly_Spend"], errors="coerce").astype(float)
    for items in LIKERT_MAP.values():
        for col in items:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(1,5).round().astype("Int64")
    return df

def winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lo, hi)

def cronbach_alpha(df_items: pd.DataFrame) -> float:
    if df_items.shape[1] < 2:
        return np.nan
    item_vars = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    k = df_items.shape[1]
    if total_var == 0:
        return np.nan
    return float((k / (k - 1)) * (1 - (item_vars.sum() / total_var)))

def calc_composites(df: pd.DataFrame, reverse_AUT3_for_alpha: bool=False) -> Tuple[pd.DataFrame, Dict[str,float]]:
    composites = {}
    alphas = {}
    df_comp = df.copy()
    for name, items in LIKERT_MAP.items():
        available = [c for c in items if c in df.columns]
        if not available:
            continue
        block = df[available].astype(float)
        if name == "AUT" and reverse_AUT3_for_alpha and "AUT3" in available:
            block_rc = block.copy()
            block_rc["AUT3"] = 6 - block_rc["AUT3"]
            alphas[name] = cronbach_alpha(block_rc.dropna())
        else:
            alphas[name] = cronbach_alpha(block.dropna())
        composites[name] = block.mean(axis=1)
    for k, v in composites.items():
        df_comp[k] = v
    return df_comp, alphas

def anova_or_kruskal(df: pd.DataFrame, metric: str, group: str="User_Type") -> Tuple[str, float]:
    if metric not in df.columns or group not in df.columns:
        return ("N/A", np.nan)
    groups = []
    for _, sub in df.groupby(group):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if len(vals) >= 6:
            groups.append(vals)
    if len(groups) < 2:
        return ("N/A", np.nan)
    normal = True
    for vals in groups:
        if len(vals) < 12:
            normal = False
            break
        _, p = stats.shapiro(vals.sample(min(500, len(vals)), random_state=RANDOM_STATE))
        if p < 0.05:
            normal = False
            break
    if normal:
        _, p = stats.f_oneway(*groups)
        return ("ANOVA", float(p))
    else:
        _, p = stats.kruskal(*groups)
        return ("Kruskal-Wallis", float(p))

def explode_multiselect(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col])
    s = df[col].fillna("").astype(str)
    s = s.apply(lambda x: [i.strip() for i in x.split(",") if i.strip() != ""])
    exploded = df.loc[s.index].copy()
    exploded[col] = s
    exploded = exploded.explode(col)
    exploded = exploded[exploded[col] != ""]
    return exploded

def compute_cooccurrence(items_series: pd.Series, top_k: int = 25) -> List[Tuple[str, str, int]]:
    pairs = {}
    for entry in items_series.dropna().astype(str):
        parts = sorted(set([p.strip() for p in entry.split(",") if p.strip()]))
        for i in range(len(parts)):
            for j in range(i+1, len(parts)):
                key = (parts[i], parts[j])
                pairs[key] = pairs.get(key, 0) + 1
    return sorted([(a,b,c) for (a,b),c in pairs.items()], key=lambda x: x[2], reverse=True)[:top_k]

def fig_to_download(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=160)
    bio.seek(0)
    return bio.getvalue()

def safe_pie(ax, sizes, labels, title=""):
    if np.sum(sizes) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title(title)

# -------------------------------- #
# --------- SIDEBAR UI ----------- #
# -------------------------------- #
def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔍 Filters")
    df_f = df.copy()

    if "Gender" in df_f.columns:
        genders = ["All"] + sorted([g for g in df_f["Gender"].dropna().unique()])
        sel_g = st.sidebar.selectbox("Gender", genders, index=0)
        if sel_g != "All":
            df_f = df_f[df_f["Gender"] == sel_g]

    if "Age" in df_f.columns:
        age_min = int(pd.to_numeric(df_f["Age"], errors="coerce").min() or 21)
        age_max = int(pd.to_numeric(df_f["Age"], errors="coerce").max() or 30)
        a1, a2 = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        df_f = df_f[(pd.to_numeric(df_f["Age"], errors="coerce") >= a1) & (pd.to_numeric(df_f["Age"], errors="coerce") <= a2)]

    if "Locality" in df_f.columns:
        locs = ["All"] + sorted(df_f["Locality"].dropna().unique().tolist())
        sel_l = st.sidebar.selectbox("Locality", locs, index=0)
        if sel_l != "All":
            df_f = df_f[df_f["Locality"] == sel_l]

    if "Monthly_Income" in df_f.columns:
        incomes = ["All"] + sorted(df_f["Monthly_Income"].dropna().unique().tolist())
        sel_i = st.sidebar.selectbox("Monthly Income Band", incomes, index=0)
        if sel_i != "All":
            df_f = df_f[df_f["Monthly_Income"] == sel_i]

    if "User_Type" in df_f.columns:
        uts = ["All"] + sorted(df_f["User_Type"].dropna().unique().tolist())
        sel_u = st.sidebar.selectbox("User Type", uts, index=0)
        if sel_u != "All":
            df_f = df_f[df_f["User_Type"] == sel_u]

    return df_f

# -------------------------------- #
# ------------ TABS -------------- #
# -------------------------------- #
def tab_overview(df: pd.DataFrame):
    st.subheader("Overview")
    st.markdown("Data audit, structure, and key distributions.")

    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": df.notna().sum().values,
        "missing": df.isna().sum().values,
        "missing_%": (df.isna().mean().values*100).round(1),
    })
    st.markdown("**Data audit**")
    st.dataframe(audit, use_container_width=True)

    cols = st.columns(2)
    with cols[0]:
        if "User_Type" in df.columns and not df["User_Type"].dropna().empty:
            vc = df["User_Type"].value_counts()
            fig1, ax1 = plt.subplots(figsize=(4,4))
            safe_pie(ax1, vc.values, vc.index.tolist(), "User Type Mix")
            st.pyplot(fig1)
    with cols[1]:
        if "Gender" in df.columns and not df["Gender"].dropna().empty:
            vc = df["Gender"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(4,4))
            safe_pie(ax2, vc.values, vc.index.tolist(), "Gender Mix")
            st.pyplot(fig2)

    cols2 = st.columns(2)
    with cols2[0]:
        if "Monthly_Income" in df.columns:
            vc = df["Monthly_Income"].value_counts().sort_index()
            fig3, ax3 = plt.subplots(figsize=(6,3.2))
            ax3.bar(vc.index.astype(str), vc.values)
            ax3.set_xticklabels(vc.index.astype(str), rotation=20, ha="right")
            ax3.set_title("Income Band Distribution")
            st.pyplot(fig3)
    with cols2[1]:
        if "Education" in df.columns:
            vc = df["Education"].value_counts()
            fig4, ax4 = plt.subplots(figsize=(6,3.2))
            ax4.bar(vc.index.astype(str), vc.values)
            ax4.set_xticklabels(vc.index.astype(str), rotation=20, ha="right")
            ax4.set_title("Education Distribution")
            st.pyplot(fig4)

    if "Locality" in df.columns:
        topn = df["Locality"].value_counts().head(10)
        fig5, ax5 = plt.subplots(figsize=(8,3.2))
        ax5.bar(topn.index.astype(str), topn.values)
        ax5.set_xticklabels(topn.index.astype(str), rotation=30, ha="right")
        ax5.set_title("Top-10 Localities by Count")
        st.pyplot(fig5)

def tab_segments_personas(df: pd.DataFrame):
    st.subheader("Segments & Personas")
    if "User_Type" not in df.columns:
        st.warning("Missing `User_Type` — segment KPIs unavailable.")
        return

    g = df.groupby("User_Type")
    kpi = pd.DataFrame({
        "count": g.size(),
        "share_%": (g.size()/len(df)*100).round(1),
        "avg_Monthly_Spend": g["Monthly_Spend"].mean().round(1) if "Monthly_Spend" in df.columns else np.nan,
        "pct_satisfied": (g["Is_Satisfied"].apply(lambda s: (s=="Yes").mean())*100).round(1) if "Is_Satisfied" in df.columns else np.nan,
        "pct_recommend": (g["Recommendation"].apply(lambda s: (s=="Yes").mean())*100).round(1) if "Recommendation" in df.columns else np.nan
    })
    st.dataframe(kpi, use_container_width=True)

    if "Switching_Brands" in df.columns:
        def tab_segments_personas(df: pd.DataFrame):
    st.subheader("Segments & Personas")
    if "User_Type" not in df.columns:
        st.warning("Missing `User_Type` — segment KPIs unavailable.")
        return

    # KPI table by User_Type
    g = df.groupby("User_Type")
    kpi = pd.DataFrame({
        "count": g.size(),
        "share_%": (g.size()/len(df)*100).round(1),
        "avg_Monthly_Spend": g["Monthly_Spend"].mean().round(1) if "Monthly_Spend" in df.columns else np.nan,
        "pct_satisfied": (g["Is_Satisfied"].apply(lambda s: (s == "Yes").mean())*100).round(1) if "Is_Satisfied" in df.columns else np.nan,
        "pct_recommend": (g["Recommendation"].apply(lambda s: (s == "Yes").mean())*100).round(1) if "Recommendation" in df.columns else np.nan,
    })
    st.dataframe(kpi, use_container_width=True)

    # Themed stacked bars for Switching Behavior
    if "Switching_Brands" in df.columns:
        pivot = pd.crosstab(df["User_Type"], df["Switching_Brands"], normalize="index")

        # Light Unilever palette per switching category (fallbacks if levels missing)
        SWITCH_COLORS = {
            "Never": "#1F70C1",      # Unilever blue
            "Rarely": "#77C3FF",     # light azure
            "Sometimes": "#BBAAF7",  # soft lilac
            "Often": "#74D4B3",      # mint
        }

        fig, ax = plt.subplots(figsize=(7, 3))
        left = np.zeros(len(pivot))
        for col in pivot.columns:
            ax.bar(
                pivot.index,
                pivot[col].values,
                bottom=left,
                label=col,
                color=SWITCH_COLORS.get(col, None),
                edgecolor="white",
                linewidth=1.0,
            )
            left += pivot[col].values

        ax.set_title("Switching Behavior by Segment (row-normalized)")
        leg = ax.legend(loc="upper right", fontsize=8, frameon=True)
        if leg and leg.get_frame():
            leg.get_frame().set_alpha(0.9)
            leg.get_frame().set_edgecolor("#D6E6FF")
        st.pyplot(fig)

    # Chi-square crosstabs (helper defined at top-level indentation)
    def chi_block(col: str, title: str):
        if col in df.columns:
            ct = pd.crosstab(df["User_Type"], df[col])
            chi2, p, _, _ = stats.chi2_contingency(ct)
            st.markdown(f"**{title}** (χ²={chi2:.2f}, p={p:.4f})")
            st.dataframe(ct)

    for c, t in [
        ("Gender", "User_Type × Gender"),
        ("Monthly_Income", "User_Type × Income"),
        ("Education", "User_Type × Education"),
    ]:
        chi_block(c, t)

    # Persona cards
    st.markdown("### Persona Cards (Top Signals)")
    for seg in df["User_Type"].dropna().unique():
        sub = df[df["User_Type"] == seg]
        info = sub["Info_Source"].value_counts().head(3).index.tolist() if "Info_Source" in sub.columns else []
        brands = sub["Brand_Preference"].value_counts().head(3).index.tolist() if "Brand_Preference" in sub.columns else []
        bars = explode_multiselect(sub, "Barriers")["Barriers"].value_counts().head(3).index.tolist() if "Barriers" in sub.columns else []
        mots = explode_multiselect(sub, "Motivators")["Motivators"].value_counts().head(3).index.tolist() if "Motivators" in sub.columns else []
        st.info(
            f"**{seg}** • Info Sources: {', '.join(info) if info else 'N/A'} | "
            f"Brands: {', '.join(brands) if brands else 'N/A'} | "
            f"Barriers: {', '.join(bars) if bars else 'N/A'} | "
            f"Motivators: {', '.join(mots) if mots else 'N/A'}"
        )


    def chi_block(col: str, title: str):
        if col in df.columns:
            ct = pd.crosstab(df["User_Type"], df[col])
            chi2, p, _, _ = stats.chi2_contingency(ct)
            st.markdown(f"**{title}** (χ²={chi2:.2f}, p={p:.4f})")
            st.dataframe(ct)

    for c, t in [("Gender","User_Type × Gender"),
                 ("Monthly_Income","User_Type × Income"),
                 ("Education","User_Type × Education")]:
        chi_block(c, t)

    st.markdown("### Persona Cards (Top Signals)")
    for seg in df["User_Type"].dropna().unique():
        sub = df[df["User_Type"]==seg]
        info = sub["Info_Source"].value_counts().head(3).index.tolist() if "Info_Source" in sub.columns else []
        brands = sub["Brand_Preference"].value_counts().head(3).index.tolist() if "Brand_Preference" in sub.columns else []
        bars = explode_multiselect(sub, "Barriers")["Barriers"].value_counts().head(3).index.tolist() if "Barriers" in sub.columns else []
        mots = explode_multiselect(sub, "Motivators")["Motivators"].value_counts().head(3).index.tolist() if "Motivators" in sub.columns else []
        st.info(f"**{seg}** • Info Sources: {', '.join(info) if info else 'N/A'} | Brands: {', '.join(brands) if brands else 'N/A'} | "
                f"Barriers: {', '.join(bars) if bars else 'N/A'} | Motivators: {', '.join(mots) if mots else 'N/A'}")

def tab_tpb_sdt(df: pd.DataFrame):
    st.subheader("TPB/SDT")
    st.markdown("Composite builder, reliability (Cronbach’s α), group differences, and correlations.")

    reverse_for_alpha = st.checkbox("Reverse-code AUT3 for α computation only (recommended)", value=True)
    df2, alphas = calc_composites(df, reverse_AUT3_for_alpha=reverse_for_alpha)

    a_table = pd.DataFrame({"Construct": list(alphas.keys()), "Cronbach_alpha": [round(alphas[k],3) for k in alphas]})
    st.dataframe(a_table, use_container_width=True)

    if "User_Type" in df2.columns:
        comps = [c for c in ["ATT","SN","PBC","BI","AUT","COMP","REL"] if c in df2.columns]
        for c in comps:
            fig, ax = plt.subplots(figsize=(6,3))
            data = [pd.to_numeric(df2[df2["User_Type"]==g][c], errors="coerce").dropna().values
                    for g in df2["User_Type"].dropna().unique()]
            labels = df2["User_Type"].dropna().unique().tolist()
            try:
                ax.violinplot(data, showmeans=True, showmedians=False)
                ax.set_xticks(np.arange(1, len(labels)+1))
                ax.set_xticklabels(labels)
                ax.set_title(f"{c} by User_Type")
            except Exception:
                ax.boxplot(data)
                ax.set_xticks(np.arange(1, len(labels)+1))
                ax.set_xticklabels(labels)
                ax.set_title(f"{c} by User_Type")
            st.pyplot(fig)

            test_name, pval = anova_or_kruskal(df2, c, "User_Type")
            st.caption(f"Test: {test_name} • p={pval:.4f}" if not np.isnan(pval) else "Insufficient data for test.")

    comps_all = [c for c in ["ATT","SN","PBC","BI","AUT","COMP","REL"] if c in df2.columns]
    if comps_all:
        corr = df2[comps_all].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(comps_all)))
        ax.set_yticks(range(len(comps_all)))
        ax.set_xticklabels(comps_all, rotation=30, ha="right")
        ax.set_yticklabels(comps_all)
        ax.set_title("Spearman Correlation (Composites)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

def tab_barriers_motivators(df: pd.DataFrame):
    st.subheader("Barriers & Motivators")
    eb = explode_multiselect(df, "Barriers")
    em = explode_multiselect(df, "Motivators")

    cols = st.columns(2)
    with cols[0]:
        if not eb.empty:
            top_b = eb["Barriers"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(top_b.index.astype(str), top_b.values)
            ax.set_xticklabels(top_b.index.astype(str), rotation=30, ha="right")
            ax.set_title("Top Barriers")
            st.pyplot(fig)
        else:
            st.info("No Barriers data to display.")

    with cols[1]:
        if not em.empty:
            top_m = em["Motivators"].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(top_m.index.astype(str), top_m.values)
            ax.set_xticklabels(top_m.index.astype(str), rotation=30, ha="right")
            ax.set_title("Top Motivators")
            st.pyplot(fig)
        else:
            st.info("No Motivators data to display.")

    for title, series in [("Barrier Co-occurrences", df["Barriers"] if "Barriers" in df.columns else pd.Series(dtype=str)),
                          ("Motivator Co-occurrences", df["Motivators"] if "Motivators" in df.columns else pd.Series(dtype=str))]:
        pairs = compute_cooccurrence(series, top_k=25)
        if not pairs:
            st.info(f"No co-occurrence pairs for {title}.")
            continue
        G = nx.Graph()
        for a,b,w in pairs:
            G.add_edge(a,b,weight=int(w))
        pos = nx.spring_layout(G, seed=RANDOM_STATE, k=0.9)
        fig, ax = plt.subplots(figsize=(6,4))
        weights = [G[u][v]['weight'] for u,v in G.edges()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        nx.draw_networkx_edges(G, pos, ax=ax, width=[0.5 + 0.3*w for w in weights])
        ax.set_title(title)
        ax.axis("off")
        st.pyplot(fig)

    df2, _ = calc_composites(df, reverse_AUT3_for_alpha=True)
    if "BI" in df2.columns:
        st.markdown("**What would move the needle (BI predictors, Spearman ρ)**")
        preds = [c for c in ["ATT","SN","PBC","AUT","COMP","REL"] if c in df2.columns]
        if preds:
            corrs = pd.Series({p: df2[[p,"BI"]].corr(method="spearman").iloc[0,1] for p in preds})
            st.dataframe(corrs.sort_values(ascending=False).round(3))

def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df2, _ = calc_composites(df, reverse_AUT3_for_alpha=True)
    if "User_Type" not in df2.columns:
        return pd.DataFrame(), pd.Series(dtype=int), [], []
    y = (df2["User_Type"] == "Frequent").astype(int)
    numeric = [c for c in ["ATT","SN","PBC","BI","AUT","COMP","REL","Age"] if c in df2.columns]
    cat = []
    if "Gender" in df2.columns: cat.append("Gender")
    if "Monthly_Income" in df2.columns: cat.append("Monthly_Income")
    if "Education" in df2.columns: cat.append("Education")
    X = df2[numeric + cat].copy()
    return X, y, numeric, cat

def _prepare_regression(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df2, _ = calc_composites(df, reverse_AUT3_for_alpha=True)
    if "Monthly_Spend" not in df2.columns:
        return pd.DataFrame(), pd.Series(dtype=float), [], []
    sub = df2[df2["Monthly_Spend"] > 0].copy()
    if sub.empty:
        return pd.DataFrame(), pd.Series(dtype=float), [], []
    y = winsorize_series(sub["Monthly_Spend"].astype(float))
    numeric = [c for c in ["ATT","SN","PBC","BI","AUT","COMP","REL","Age"] if c in sub.columns]
    cat = []
    if "Gender" in sub.columns: cat.append("Gender")
    if "Monthly_Income" in sub.columns: cat.append("Monthly_Income")
    if "Education" in sub.columns: cat.append("Education")
    X = sub[numeric + cat].copy()
    return X, y, numeric, cat

def plot_confusion(cm: np.ndarray, labels=("Non-Frequent","Frequent")):
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z}", ha='center', va='center')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def tab_predict_explain(df: pd.DataFrame):
    st.subheader("Predict & Explain")
    X, y, num_cols, cat_cols = _prepare_features(df)
    if not X.empty:
        st.markdown("**Classification: Frequent vs Others**")
        folds = 5 if len(y) >= 80 else 3
        class_weight = "balanced" if y.value_counts().min() / max(1, y.value_counts().max()) < 0.6 else None

        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=False), [c for c in num_cols if c in X.columns]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cat_cols if c in X.columns])
            ],
            remainder="drop"
        )
        clf = Pipeline(steps=[
            ("prep", pre),
            ("clf", LogisticRegression(solver="liblinear", class_weight=class_weight, random_state=RANDOM_STATE))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:,1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        }
        st.dataframe(pd.Series(metrics).round(3))

        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_validate(clf, X, y, scoring=["accuracy","precision","recall","f1","roc_auc"], cv=cv)
        st.caption("Cross-Validation (mean±std)")
        st.dataframe(pd.Series({k.replace("test_",""): f"{np.mean(v):.3f}±{np.std(v):.3f}" for k,v in cv_scores.items() if k.startswith("test_")}))

        cm = confusion_matrix(y_test, y_pred)
        st.pyplot(plot_confusion(cm))

        try:
            result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
            importances = result.importances_mean
            feat_names = num_cols + cat_cols
            imp = pd.Series(importances[:len(feat_names)], index=feat_names).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(imp.index[:10], np.abs(imp.values[:10]))
            ax.set_xticklabels(imp.index[:10], rotation=30, ha="right")
            ax.set_title("Permutation Importance (top 10, abs)")
            st.pyplot(fig)
        except Exception as e:
            st.info(f"Permutation importance not available: {e}")

        top_for_pdp = [c for c in ["ATT","SN","PBC","BI"] if c in num_cols][:3]
        if top_for_pdp:
            for feat in top_for_pdp:
                try:
                    fig, ax = plt.subplots(figsize=(5,3))
                    PartialDependenceDisplay.from_estimator(
                        clf, X_test, [feat], kind="average", ax=ax,
                        response_method="predict_proba", grid_resolution=20
                    )
                    ax.set_title(f"Partial Dependence: {feat} → P(Frequent)")
                    st.pyplot(fig)
                except Exception as e:
                    st.info(f"PDP for {feat} unavailable: {e}")
        st.divider()
    else:
        st.info("Classification block hidden (insufficient columns).")

    Xr, yr, num_r, cat_r = _prepare_regression(df)
    if not Xr.empty:
        st.markdown("**Regression: Monthly_Spend (users only)**")
        pre_r = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=False), [c for c in num_r if c in Xr.columns]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cat_r if c in Xr.columns])
            ],
            remainder="drop"
        )
        rf = Pipeline(steps=[
            ("prep", pre_r),
            ("rf", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE))
        ])
        X_train, X_test, y_train, y_test = train_test_split(Xr, yr, test_size=0.25, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        y_hat = rf.predict(X_test)

        mse_val = mean_squared_error(y_test, y_hat)
        rmse_val = float(np.sqrt(mse_val))
        metrics_r = {
            "RMSE": rmse_val,
            "MAE": mean_absolute_error(y_test, y_hat),
            "R2": r2_score(y_test, y_hat)
        }
        st.dataframe(pd.Series(metrics_r).round(2))

        cv = 5 if len(yr) >= 80 else 3
        rmse = -cross_val_score(rf, Xr, yr, scoring="neg_root_mean_squared_error", cv=cv)
        r2 = cross_val_score(rf, Xr, yr, scoring="r2", cv=cv)
        st.caption("Cross-Validation (RMSE, R²)")
        st.write(f"RMSE: {rmse.mean():.2f}±{rmse.std():.2f} | R²: {r2.mean():.3f}±{r2.std():.3f}")
    else:
        st.info("Regression block hidden (Monthly_Spend missing or all zeros).")

def tab_reco_roi(df: pd.DataFrame):
    st.subheader("Recommendations & ROI Sandbox")
    df2, _ = calc_composites(df, reverse_AUT3_for_alpha=True)

    st.markdown("**Simulate composite lifts (mean shifts added before scoring)**")
    deltas = {}
    for c in ["ATT","SN","PBC","AUT","COMP","REL"]:
        if c in df2.columns:
            deltas[c] = st.slider(f"Δ{c}", -1.0, 1.0, 0.0, 0.1)
    barrier_reduct = st.slider("Barrier reduction (%) — conceptual (affects narrative KPIs only)", 0, 50, 0, 5)

    X, y, num_cols, cat_cols = _prepare_features(df2)
    if not X.empty:
        X_adj = X.copy()
        for c, d in deltas.items():
            if c in X_adj.columns:
                X_adj[c] = (X_adj[c] + d).clip(1,5)

        pre = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=False), [c for c in num_cols if c in X.columns]),
                ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cat_cols if c in X.columns])
            ],
            remainder="drop"
        )
        clf = Pipeline(steps=[
            ("prep", pre),
            ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE))
        ])
        clf.fit(X, y)
        base_prob = clf.predict_proba(X)[:,1]
        new_prob  = clf.predict_proba(X_adj)[:,1]

        base_freq = (base_prob >= 0.5).mean()
        new_freq  = (new_prob  >= 0.5).mean()

        if "Recommendation" in df2.columns:
            rec_rate = (df2["Recommendation"] == "Yes").mean()
            avg_delta = np.mean([deltas.get(k,0) for k in ["ATT","SN","PBC"] if k in X.columns]) if ["ATT","SN","PBC"] else 0
            rec_uplift = max(0.0, min(0.1, 0.02 + 0.05*avg_delta))
            new_rec = min(1.0, rec_rate + rec_uplift)
        else:
            new_rec = np.nan
            rec_rate = np.nan

        if "Monthly_Spend" in df2.columns and df2["Monthly_Spend"].fillna(0).sum() > 0:
            Xr, yr, num_r, cat_r = _prepare_regression(df2)
            if not Xr.empty:
                pre_r = ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(with_mean=False), [c for c in num_r if c in Xr.columns]),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in cat_r if c in Xr.columns])
                    ], remainder="drop"
                )
                rf = Pipeline(steps=[("prep", pre_r), ("rf", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE))])
                rf.fit(Xr, yr)
                Xr_adj = Xr.copy()
                for c, d in deltas.items():
                    if c in Xr_adj.columns:
                        Xr_adj[c] = (Xr_adj[c] + d).clip(1,5)
                spend_base = rf.predict(Xr).mean()
                spend_new  = rf.predict(Xr_adj).mean()
            else:
                spend_base = spend_new = np.nan
        else:
            spend_base = spend_new = np.nan

        st.markdown("### Uplift Results")
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Frequent Share (pred.)", f"{new_freq*100:,.1f}%", delta=f"{(new_freq-base_freq)*100:,.1f} pp")
        with colB:
            if not np.isnan(new_rec):
                st.metric("Recommendation Rate (proxy)", f"{new_rec*100:,.1f}%", delta=f"{(new_rec-rec_rate)*100:,.1f} pp")
            else:
                st.metric("Recommendation Rate (proxy)", "N/A")
        with colC:
            if not np.isnan(spend_new):
                st.metric("Avg Monthly Spend (₹, users)", f"{spend_new:,.0f}", delta=f"{(spend_new-spend_base):,.0f}")
            else:
                st.metric("Avg Monthly Spend (₹, users)", "N/A")

        st.caption(f"Barrier reduction set to {barrier_reduct}%: treat as scenario annotation in your slide narrative.")

        out = df2.copy()
        out["P_Frequent_Base"] = base_prob
        out["P_Frequent_New"]  = new_prob
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Scored CSV (base vs scenario)", data=csv_bytes, file_name="scored_uplift.csv", mime="text/csv")
    else:
        st.info("ROI sandbox needs composites + demographics + User_Type.")

# -------------------------------- #
# ------------- MAIN ------------- #
# -------------------------------- #
def main():
    set_page()

    st.sidebar.header("📂 Data")
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    df_raw = load_csv(up)
    if df_raw.empty:
        st.error("No data available. Please upload `skincare_survey_mumbai_120.csv` or place it in ./data/.")
        return

    df = strip_whitespace_and_cast(df_raw)
    df_f = sidebar_filters(df)

    tabs = st.tabs(["Overview", "Segments & Personas", "TPB/SDT", "Barriers & Motivators", "Predict & Explain", "Recommendations & ROI"])
    with tabs[0]:
        tab_overview(df_f)
    with tabs[1]:
        tab_segments_personas(df_f)
    with tabs[2]:
        tab_tpb_sdt(df_f)
    with tabs[3]:
        tab_barriers_motivators(df_f)
    with tabs[4]:
        tab_predict_explain(df_f)
    with tabs[5]:
        tab_reco_roi(df_f)

    df2, _ = calc_composites(df_f, reverse_AUT3_for_alpha=True)
    comps_all = [c for c in ["ATT","SN","PBC","BI","AUT","COMP","REL"] if c in df2.columns]
    if comps_all:
        corr = df2[comps_all].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(5,4))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(comps_all)))
        ax.set_yticks(range(len(comps_all)))
        ax.set_xticklabels(comps_all, rotation=30, ha="right")
        ax.set_yticklabels(comps_all)
        ax.set_title("Spearman Correlation (Composites)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        png = fig_to_download(fig)
        st.download_button("⬇ Download Correlation Heatmap (PNG)", data=png, file_name="correlation_heatmap.png", mime="image/png")

    st.caption("Built with Streamlit • Matplotlib • scikit-learn • SciPy • NetworkX | All charts seaborn-free.")

if __name__ == "__main__":
    main()
