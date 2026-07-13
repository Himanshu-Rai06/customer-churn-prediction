import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, roc_auc_score
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")  # non-interactive backend for Streamlit

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SVG ICONS (replace emoji usage throughout)
# ─────────────────────────────────────────────
def icon(name, size=16, color="currentColor"):
    """Return an inline SVG icon as an HTML string."""
    icons = {
        "bank": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 21h18"/><path d="M4 21V10"/><path d="M20 21V10"/>
            <path d="M2 10l10-6 10 6"/><path d="M8 21v-6"/><path d="M12 21v-6"/><path d="M16 21v-6"/>
        </svg>''',
        "chart-bar": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/>
            <rect x="12" y="8" width="3" height="10"/><rect x="17" y="5" width="3" height="13"/>
        </svg>''',
        "trend-up": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 17 9 11 13 15 21 7"/><polyline points="14 7 21 7 21 14"/>
        </svg>''',
        "crystal-ball": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="10" r="7"/><path d="M6 21h12"/><path d="M9 21l1-4"/><path d="M15 21l-1-4"/>
        </svg>''',
        "calculator": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <rect x="4" y="2" width="16" height="20" rx="2"/><line x1="8" y1="6" x2="16" y2="6"/>
            <line x1="8" y1="10" x2="8" y2="10"/><line x1="12" y1="10" x2="12" y2="10"/>
            <line x1="16" y1="10" x2="16" y2="10"/><line x1="8" y1="14" x2="8" y2="14"/>
            <line x1="12" y1="14" x2="12" y2="14"/><line x1="16" y1="14" x2="16" y2="14"/>
            <line x1="8" y1="18" x2="8" y2="18"/><line x1="12" y1="18" x2="12" y2="18"/>
            <line x1="16" y1="18" x2="16" y2="18"/>
        </svg>''',
        "person": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/>
        </svg>''',
        "wallet": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 7a2 2 0 0 1 2-2h13a1 1 0 0 1 1 1v3"/>
            <path d="M3 7v11a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-8a2 2 0 0 0-2-2H5"/>
            <circle cx="17" cy="14" r="1.4"/>
        </svg>''',
        "handshake": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 12l5-4 4 3 3-3 5 4"/><path d="M7 8v8"/><path d="M17 8v8"/>
            <path d="M9 11l3 3 3-3"/>
        </svg>''',
        "circle-check": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.5 2.5 4.5-5"/>
        </svg>''',
        "circle-alert": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12" y2="13"/>
            <line x1="12" y1="16" x2="12" y2="16"/>
        </svg>''',
        "circle-info": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/>
            <line x1="12" y1="8" x2="12" y2="8"/>
        </svg>''',
        "dot": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24"><circle cx="12" cy="12" r="7" fill="{color}"/></svg>''',
        "dial": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"
            viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round">
            <path d="M4 15a8 8 0 0 1 16 0"/><path d="M12 15l4-5"/><circle cx="12" cy="15" r="1"/>
        </svg>''',
    }
    return icons.get(name, "")

def risk_dot(color, size=11):
    return f'<span style="display:inline-block;width:{size}px;height:{size}px;border-radius:50%;background:{color};margin-right:6px;vertical-align:middle;"></span>'

# ─────────────────────────────────────────────
# GLOBAL STYLE POLISH
# (config.toml handles the base theme — this layers in font,
#  card shadows, and small details the theme alone can't reach)
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    /* card-like bordered containers get a soft shadow + tighter radius */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        transition: box-shadow 0.2s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }

    /* tighten block spacing so panels feel compact, not airy */
    div[data-testid="stVerticalBlock"] { gap: 0.6rem; }

    /* metric labels: smaller + uppercase for a "dashboard" feel */
    div[data-testid="stMetricLabel"] {
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.75rem !important;
        opacity: 0.75;
    }

    section[data-testid="stSidebar"] h1 { line-height: 1.2; }

    /* refined sidebar navigation — turn "N · Label" radio options into
       a clean numbered list instead of emoji-style circled digits */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 6px 4px;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 0.92rem !important;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
THRESHOLD = 0.206
FEATURE_COLS = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
    "Geography_Germany", "Geography_Spain",
    "Balance_to_Salary", "Products_per_Year",
    "Age_Tenure_Interaction", "Engagement_Score",
]

NAV_OPTIONS = [
    "1 · Risk Calculator",
    "2 · Probability Distribution",
    "3 · Feature Importance",
    "4 · What-If Simulator",
]

# ─────────────────────────────────────────────
# LOAD RESOURCES  (cached so they load once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("model/final_model_gradient_boosting.pkl")

@st.cache_data(show_spinner=False)
def load_data():
    raw = pd.read_csv("data/European_Bank.csv")
    model_df = pd.read_csv("data/df_model_cleaned.csv")
    return raw, model_df

@st.cache_data
def get_test_predictions(_model, model_df):
    """Reproduce the same test split used in training, return predictions."""
    X = model_df[FEATURE_COLS]
    y = model_df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_proba = _model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= THRESHOLD).astype(int)
    return X_test, y_test, y_proba, y_pred

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def get_shap_values(_model_key, model_df):
    """Cache SHAP values for the full test set."""
    model  = load_model()
    X      = model_df[FEATURE_COLS]
    y      = model_df["Exited"]
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values, X_test, explainer.expected_value


# ─────────────────────────────────────────────
# HELPER: build engineered features from raw inputs
# ─────────────────────────────────────────────
def engineer_features(
    credit_score, gender, age, tenure, balance,
    num_products, has_cr_card, is_active, salary,
    geography
):
    """
    Mirror the exact feature engineering done in the Colab notebook.
    Returns a single-row DataFrame in the correct column order.
    """
    CAP_99 = 35.476218          # 99th-percentile cap used during training

    geo_germany = 1 if geography == "Germany" else 0
    geo_spain   = 1 if geography == "Spain"   else 0
    gender_enc  = 1 if gender    == "Male"    else 0

    balance_to_salary      = min(balance / (salary + 1), CAP_99)
    products_per_year      = num_products / (tenure + 1)
    age_tenure_interaction = age * tenure
    engagement_score       = is_active * num_products

    row = {
        "CreditScore":           credit_score,
        "Gender":                gender_enc,
        "Age":                   age,
        "Tenure":                tenure,
        "Balance":               balance,
        "NumOfProducts":         num_products,
        "HasCrCard":             has_cr_card,
        "IsActiveMember":        is_active,
        "EstimatedSalary":       salary,
        "Geography_Germany":     geo_germany,
        "Geography_Spain":       geo_spain,
        "Balance_to_Salary":     balance_to_salary,
        "Products_per_Year":     products_per_year,
        "Age_Tenure_Interaction":age_tenure_interaction,
        "Engagement_Score":      engagement_score,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ─────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────
def risk_colour(prob):
    if prob >= 0.6:
        return "#D85A30"   # red
    elif prob >= THRESHOLD:
        return "#BA7517"   # amber
    else:
        return "#1D9E75"   # green

def risk_label(prob):
    if prob >= 0.6:
        return "HIGH RISK"
    elif prob >= THRESHOLD:
        return "MODERATE RISK"
    else:
        return "LOW RISK"


# ─────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────
try:
    with st.spinner("Loading model & data…"):
        model            = load_model()
        raw_df, model_df = load_data()
        X_test, y_test, y_proba_test, y_pred_test = get_test_predictions(model, model_df)
        shap_vals, X_test_shap, expected_value     = get_shap_values("gb", model_df)
except Exception as e:
    load_model.clear()
    load_data.clear()
    st.error(
        "Couldn't load the model or dataset. Please check that the "
        "`model/` and `data/` folders are present, then refresh the page."
    )
    st.caption(f"Details: {e}")
    st.stop()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;'>"
        f"{icon('bank', 30, '#7F77DD')}"
        f"<span style='font-size:1.25rem;font-weight:700;'>Bank Churn Dashboard</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("Customer retention intelligence")
    st.divider()

    page = st.radio(
        "Navigate",
        NAV_OPTIONS,
        label_visibility="collapsed",
    )

    st.divider()
    with st.container(border=True):
        st.markdown(
            f"""
            <div style='font-size:0.82rem; line-height:1.7'>
            <b>Model</b> &nbsp;Gradient Boosting<br>
            <b>Threshold</b> &nbsp;{THRESHOLD}<br>
            <b>ROC-AUC</b> &nbsp;0.867<br>
            <b>Dataset</b> &nbsp;10,000 customers
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# MODULE 1 — RISK CALCULATOR
# ══════════════════════════════════════════════
if page == "1 · Risk Calculator":
    st.markdown(
        f"<h4 style='display:flex;align-items:center;gap:10px;'>"
        f"{icon('calculator', 22, '#7F77DD')} Customer Churn Risk Calculator</h4>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Edit a customer's profile on the left — the risk score and driver "
        "breakdown on the right update instantly."
    )

    # ── 1:3 LAYOUT — inputs | results ────────────
    panel_col, results_col = st.columns([1, 3], gap="medium")

    # ── LEFT (1) — compact input panel ───────────
    with panel_col:
        with st.container(border=True, height="stretch"):
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:7px;font-weight:600;'>"
                f"{icon('person', 16)} Demographics</div>",
                unsafe_allow_html=True,
            )
            age       = st.slider("Age", 18, 92, 38)
            gender    = st.selectbox("Gender", ["Female", "Male"])
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:7px;font-weight:600;'>"
                f"{icon('wallet', 16)} Account</div>",
                unsafe_allow_html=True,
            )
            credit_score = st.slider("Credit score", 350, 850, 650)
            balance = st.number_input(
                "Balance (€)", 0.0, 250900.0, 76000.0, step=500.0,
                format="%.0f",
            )
            salary = st.number_input(
                "Salary (€)", 11.0, 200000.0, 100000.0, step=500.0,
                format="%.0f",
            )
            tenure = st.slider("Tenure (yrs)", 0, 10, 5)

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:7px;font-weight:600;'>"
                f"{icon('handshake', 16)} Engagement</div>",
                unsafe_allow_html=True,
            )
            num_products = st.selectbox("Products", [1, 2, 3, 4], index=0)
            has_cr_card  = st.toggle("Has credit card", value=True)
            is_active    = st.toggle("Active member", value=True)

    # ── PREDICTION ────────────────────────────────
    input_df = engineer_features(
        credit_score, gender, age, tenure, balance,
        num_products, int(has_cr_card), int(is_active), salary, geography
    )
    prob = model.predict_proba(input_df)[0, 1]

    # ── RIGHT (3) — results + explanation ────────
    with results_col:
        with st.container(border=True):
            r1, r2, r3 = st.columns([1, 1, 2], vertical_alignment="center")

            with r1:
                st.metric("Churn probability", f"{prob:.1%}")

            with r2:
                colour = risk_colour(prob)
                label  = risk_label(prob)
                st.markdown(
                    f"""
                    <div style='padding:12px 16px;border-radius:10px;
                                background:{colour}22;border:1.5px solid {colour};
                                font-size:15px;font-weight:600;color:{colour};
                                text-align:center'>
                        {risk_dot(colour)}{label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r3:
                fig_bar, ax_bar = plt.subplots(figsize=(4, 0.45))
                ax_bar.barh(0, prob, color=risk_colour(prob), height=0.45)
                ax_bar.barh(0, 1 - prob, left=prob, color="#3A3A4A", height=0.45)
                ax_bar.axvline(THRESHOLD, color="white", lw=1.3, ls="--")
                ax_bar.set_xlim(0, 1)
                ax_bar.set_yticks([])
                ax_bar.set_xticks([])
                ax_bar.text(
                    THRESHOLD + 0.015, 0, f"threshold {THRESHOLD}",
                    fontsize=7.5, va="center", color="white"
                )
                for spine in ax_bar.spines.values():
                    spine.set_visible(False)
                fig_bar.patch.set_alpha(0)
                ax_bar.patch.set_alpha(0)
                fig_bar.tight_layout(pad=0.2)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)

        with st.container(border=True, height="stretch"):
            st.markdown("**Why did the model predict this?**")
            explainer_live = get_shap_explainer(model)
            sv_single      = explainer_live.shap_values(input_df)

            fig_wf, ax_wf = plt.subplots(figsize=(9, 4.2))
            shap.waterfall_plot(
                shap.Explanation(
                    values       = sv_single[0],
                    base_values  = float(np.array(explainer_live.expected_value).flat[0]),
                    data         = input_df.iloc[0],
                    feature_names= FEATURE_COLS,
                ),
                show=False,
            )
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close("all")

            st.caption(
                "Each bar shows how much a feature **pushed** the prediction "
                "toward churn (+) or away from churn (−). "
                f"Baseline E[f(X)] = {float(np.array(explainer_live.expected_value).flat[0]):.3f}, "
                "the model's average output across all customers."
            )


# ══════════════════════════════════════════════
# MODULE 2 — PROBABILITY DISTRIBUTION
# ══════════════════════════════════════════════
elif page == "2 · Probability Distribution":
    st.markdown(
        f"<h2 style='display:flex;align-items:center;gap:10px;'>"
        f"{icon('chart-bar', 26, '#7F77DD')} Churn Probability Distribution</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "How the model's predicted probabilities are distributed across the "
        "held-out test set (2,000 customers), and how the chosen threshold "
        "determines the precision/recall trade-off."
    )
    st.divider()

    # ── KPI STRIP ───────────────────────────────
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    with st.container(border=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("ROC-AUC",  "0.867")
        k2.metric("Recall",   f"{tp/(tp+fn):.1%}", help="Of all actual churners, % correctly flagged")
        k3.metric("Precision",f"{tp/(tp+fp):.1%}", help="Of all flagged, % who actually churned")
        k4.metric("Churners caught",  f"{tp} / {tp+fn}")
        k5.metric("False alarms",     str(fp))

    col_l, col_r = st.columns(2)

    # ── PROBABILITY HISTOGRAM ────────────────────
    with col_l, st.container(border=True, height="stretch"):
        st.subheader("Predicted probability by actual label")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
        ax_hist.hist(
            y_proba_test[y_test == 0], bins=40, alpha=0.6,
            color="#378ADD", label="Retained (Exited=0)", density=True
        )
        ax_hist.hist(
            y_proba_test[y_test == 1], bins=40, alpha=0.6,
            color="#D85A30", label="Churned  (Exited=1)", density=True
        )
        ax_hist.axvline(
            THRESHOLD, color="black", lw=1.5, ls="--",
            label=f"Threshold ({THRESHOLD})"
        )
        ax_hist.set_xlabel("Predicted churn probability")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=8)
        ax_hist.set_title("Probability distribution by true label")
        fig_hist.tight_layout()
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

        st.caption(
            "Blue = customers who stayed, Red = customers who churned. "
            "A well-calibrated model separates these two distributions. "
            "The dashed line shows where we draw the churn/no-churn boundary."
        )

    # ── CONFUSION MATRIX ────────────────────────
    with col_r, st.container(border=True, height="stretch"):
        st.subheader("Confusion matrix at threshold 0.206")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: Stay", "Pred: Churn"],
            yticklabels=["True: Stay", "True: Churn"],
            ax=ax_cm, linewidths=0.5,
        )
        ax_cm.set_title("Confusion Matrix (threshold = 0.206)")
        fig_cm.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

        st.caption(
            f"**{tp}** churners correctly flagged · "
            f"**{fn}** missed · "
            f"**{fp}** false alarms · "
            f"**{tn}** correctly cleared"
        )

    st.divider()

    # ── THRESHOLD EXPLORER ───────────────────────
    with st.container(border=True):
        st.markdown(
            f"<h3 style='display:flex;align-items:center;gap:9px;'>"
            f"{icon('dial', 20, '#7F77DD')} Threshold explorer</h3>",
            unsafe_allow_html=True,
        )
        st.caption("Drag the slider to see how precision and recall change at different thresholds.")

        t_val = st.slider(
            "Decision threshold", 0.05, 0.95, THRESHOLD, 0.01,
            help="The model flags a customer as 'churn risk' if their probability ≥ this value."
        )
        y_pred_t = (y_proba_test >= t_val).astype(int)
        cm_t     = confusion_matrix(y_test, y_pred_t)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall",    f"{tp_t/(tp_t+fn_t):.1%}")
        c2.metric("Precision", f"{tp_t/(tp_t+fp_t):.1%}" if (tp_t+fp_t) > 0 else "N/A")
        c3.metric("Churners caught", f"{tp_t} / {tp_t+fn_t}")
        c4.metric("False alarms",    str(fp_t))


# ══════════════════════════════════════════════
# MODULE 3 — FEATURE IMPORTANCE DASHBOARD
# ══════════════════════════════════════════════
elif page == "3 · Feature Importance":
    st.markdown(
        f"<h2 style='display:flex;align-items:center;gap:10px;'>"
        f"{icon('trend-up', 26, '#7F77DD')} Feature Importance Dashboard</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Three independent views of what drives churn — raw data patterns, "
        "model-based importance, and SHAP explainability — all pointing to the same answer."
    )
    st.divider()

    tab1, tab2, tab3 = st.tabs(["EDA Insights", "Model Importance", "SHAP Analysis"])

    # ── TAB 1: EDA ──────────────────────────────
    with tab1:
        st.subheader("Churn rate by key segments")

        e1, e2 = st.columns(2)

        with e1, st.container(border=True):
            # Churn by Geography
            churn_geo = raw_df.groupby("Geography")["Exited"].mean().sort_values()
            fig_geo, ax_geo = plt.subplots(figsize=(5, 3))
            bars = ax_geo.barh(
                churn_geo.index, churn_geo.values,
                color=["#378ADD", "#1D9E75", "#D85A30"]
            )
            ax_geo.set_xlabel("Churn Rate")
            ax_geo.set_title("Churn Rate by Geography")
            for bar, val in zip(bars, churn_geo.values):
                ax_geo.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                            f"{val:.1%}", va="center", fontsize=9)
            fig_geo.tight_layout()
            st.pyplot(fig_geo, use_container_width=True)
            plt.close(fig_geo)

            # Churn by Gender
            churn_gen = raw_df.groupby("Gender")["Exited"].mean().sort_values()
            fig_gen, ax_gen = plt.subplots(figsize=(5, 2.5))
            bars2 = ax_gen.barh(
                churn_gen.index, churn_gen.values,
                color=["#378ADD", "#D85A30"]
            )
            ax_gen.set_xlabel("Churn Rate")
            ax_gen.set_title("Churn Rate by Gender")
            for bar, val in zip(bars2, churn_gen.values):
                ax_gen.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                            f"{val:.1%}", va="center", fontsize=9)
            fig_gen.tight_layout()
            st.pyplot(fig_gen, use_container_width=True)
            plt.close(fig_gen)

        with e2, st.container(border=True):
            # Churn by NumOfProducts
            churn_prod = raw_df.groupby("NumOfProducts")["Exited"].mean()
            fig_prod, ax_prod = plt.subplots(figsize=(5, 3))
            colors_prod = ["#1D9E75", "#378ADD", "#BA7517", "#D85A30"]
            bars3 = ax_prod.bar(
                churn_prod.index.astype(str), churn_prod.values,
                color=colors_prod
            )
            ax_prod.set_xlabel("Number of Products")
            ax_prod.set_ylabel("Churn Rate")
            ax_prod.set_title("Churn Rate by Number of Products")
            ax_prod.set_ylim(0, 1.1)
            for bar, val in zip(bars3, churn_prod.values):
                ax_prod.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                             f"{val:.1%}", ha="center", fontsize=9)
            fig_prod.tight_layout()
            st.pyplot(fig_prod, use_container_width=True)
            plt.close(fig_prod)

            # Churn by IsActiveMember
            churn_act = raw_df.groupby("IsActiveMember")["Exited"].mean()
            fig_act, ax_act = plt.subplots(figsize=(5, 2.5))
            bars4 = ax_act.bar(
                ["Inactive", "Active"], churn_act.values,
                color=["#D85A30", "#1D9E75"]
            )
            ax_act.set_ylabel("Churn Rate")
            ax_act.set_title("Churn Rate by Active Membership")
            ax_act.set_ylim(0, 0.40)
            for bar, val in zip(bars4, churn_act.values):
                ax_act.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                            f"{val:.1%}", ha="center", fontsize=9)
            fig_act.tight_layout()
            st.pyplot(fig_act, use_container_width=True)
            plt.close(fig_act)

        st.divider()
        with st.container(border=True):
            st.subheader("Age distribution by churn outcome")
            fig_age, ax_age = plt.subplots(figsize=(10, 3.5))
            ax_age.hist(
                raw_df[raw_df["Exited"] == 0]["Age"], bins=35,
                alpha=0.6, color="#378ADD", label="Retained", density=True
            )
            ax_age.hist(
                raw_df[raw_df["Exited"] == 1]["Age"], bins=35,
                alpha=0.6, color="#D85A30", label="Churned", density=True
            )
            ax_age.set_xlabel("Age")
            ax_age.set_ylabel("Density")
            ax_age.legend()
            ax_age.set_title("Age Distribution by Churn Outcome")
            fig_age.tight_layout()
            st.pyplot(fig_age, use_container_width=True)
            plt.close(fig_age)

            st.caption(
                "Churned customers cluster in the 40–55 age range, "
                "while retained customers are more evenly spread across younger ages."
            )

    # ── TAB 2: MODEL IMPORTANCE ──────────────────
    with tab2:
        with st.container(border=True):
            st.subheader("Gradient Boosting — feature importances")
            importances = model.feature_importances_
            feat_imp_df = (
                pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances})
                .sort_values("Importance", ascending=True)
            )

            fig_imp, ax_imp = plt.subplots(figsize=(7, 6))
            ax_imp.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color="#7F77DD")
            ax_imp.set_xlabel("Feature Importance (mean decrease in impurity)")
            ax_imp.set_title("Gradient Boosting: Feature Importances")
            fig_imp.tight_layout()
            st.pyplot(fig_imp, use_container_width=True)
            plt.close(fig_imp)

            st.caption(
                "NumOfProducts and Age dominate — consistent across all three "
                "analyses (raw EDA, feature importance, and SHAP below)."
            )

        st.divider()
        with st.container(border=True):
            st.subheader("Full model comparison at default threshold (0.5)")
            comparison_data = {
                "Model":      ["Logistic Regression", "Logistic (balanced)",
                               "Decision Tree", "Random Forest",
                               "Gradient Boosting", "XGBoost"],
                "Precision":  [0.58, 0.38, 0.44, 0.55, 0.76, 0.52],
                "Recall":     [0.19, 0.71, 0.77, 0.71, 0.48, 0.73],
                "F1":         [0.28, 0.50, 0.56, 0.62, 0.59, 0.61],
                "ROC-AUC":    [0.773, 0.775, 0.825, 0.866, 0.867, 0.859],
            }
            comp_df = pd.DataFrame(comparison_data).set_index("Model")
            st.dataframe(comp_df, use_container_width=True)

    # ── TAB 3: SHAP ─────────────────────────────
    with tab3:
        with st.container(border=True):
            st.subheader("SHAP Summary Plot — direction and magnitude")
            st.markdown(
                "**Red** = high feature value · **Blue/Teal** = low feature value  \n"
                "Points to the right of zero push the prediction toward churn; "
                "points to the left push away from churn."
            )

            fig_shap, ax_shap = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_vals, X_test_shap, show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close("all")

        with st.container(border=True):
            st.subheader("SHAP Bar Chart — average absolute impact")
            fig_bar2, ax_bar2 = plt.subplots(figsize=(7, 5))
            shap.summary_plot(shap_vals, X_test_shap, plot_type="bar", show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close("all")


# ══════════════════════════════════════════════
# MODULE 4 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════
elif page == "4 · What-If Simulator":
    st.markdown(
        f"<h2 style='display:flex;align-items:center;gap:10px;'>"
        f"{icon('crystal-ball', 26, '#7F77DD')} What-If Scenario Simulator</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Start with a customer profile, then adjust their engagement and "
        "product values to see how the churn risk responds in real time. "
        "Use this to explore what retention actions could bring a customer "
        "below the risk threshold."
    )
    st.divider()

    # ── BASE CUSTOMER ───────────────────────────
    st.subheader("Step 1 — Define the base customer")
    with st.container(border=True):
        base_col1, base_col2, base_col3 = st.columns(3)

        with base_col1:
            b_age       = st.slider("Age",          18, 92, 50,  key="b_age")
            b_gender    = st.selectbox("Gender",    ["Female", "Male"], key="b_gen")
            b_geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=1, key="b_geo")

        with base_col2:
            b_credit    = st.slider("Credit Score", 350, 850, 650, key="b_credit")
            b_balance   = st.number_input("Balance (€)",  0.0, 250900.0, 130000.0, step=500.0, key="b_bal")
            b_salary    = st.number_input("Salary (€)",   11.0, 200000.0, 50000.0, step=500.0, key="b_sal")
            b_tenure    = st.slider("Tenure (yrs)", 0, 10, 2, key="b_ten")

        with base_col3:
            b_products  = st.selectbox("Num Products", [1, 2, 3, 4], index=2, key="b_prod")
            b_crcard    = st.radio("Has Credit Card?", [1, 0],
                                   format_func=lambda x: "Yes" if x else "No", key="b_cc")
            b_active    = st.radio("Active Member?",   [0, 1],
                                   format_func=lambda x: "Yes" if x else "No",
                                   index=0, key="b_act")

    base_df   = engineer_features(
        b_credit, b_gender, b_age, b_tenure, b_balance,
        b_products, b_crcard, b_active, b_salary, b_geography
    )
    base_prob = model.predict_proba(base_df)[0, 1]

    st.divider()

    # ── SCENARIO ADJUSTMENTS ─────────────────────
    st.subheader("Step 2 — Adjust the retention levers")
    st.markdown(
        "These are the features a bank can realistically influence "
        "through a retention campaign."
    )

    with st.container(border=True):
        adj_col1, adj_col2 = st.columns(2)
        with adj_col1:
            adj_products = st.selectbox(
                "Number of Products (after campaign)",
                [1, 2, 3, 4],
                index=[1, 2, 3, 4].index(b_products),
                key="adj_prod",
                help="Tip: moving from 3→2 products often dramatically reduces risk"
            )
            adj_active   = st.radio(
                "Active Member? (after re-engagement)",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                index=b_active,
                key="adj_act"
            )

        with adj_col2:
            adj_balance = st.number_input(
                "Balance after change (€)",
                0.0, 250900.0, float(b_balance), step=500.0, key="adj_bal"
            )
            adj_tenure  = st.slider(
                "Tenure (simulated future)", 0, 10, b_tenure, key="adj_ten"
            )

    scenario_df   = engineer_features(
        b_credit, b_gender, b_age, adj_tenure, adj_balance,
        adj_products, b_crcard, adj_active, b_salary, b_geography
    )
    scenario_prob = model.predict_proba(scenario_df)[0, 1]
    delta         = scenario_prob - base_prob

    st.divider()

    # ── RESULTS ─────────────────────────────────
    st.subheader("Step 3 — See the impact")

    with st.container(border=True):
        res1, res2, res3 = st.columns(3, vertical_alignment="center")

        with res1:
            st.metric(
                "Base Risk",
                f"{base_prob:.1%}",
                help="Risk before any retention action"
            )
            col = risk_colour(base_prob)
            st.markdown(
                f"<div style='color:{col};font-weight:500'>{risk_dot(col)}{risk_label(base_prob)}</div>",
                unsafe_allow_html=True
            )

        with res2:
            st.metric(
                "Scenario Risk",
                f"{scenario_prob:.1%}",
                delta=f"{delta:+.1%}",
                delta_color="inverse",
            )
            col2 = risk_colour(scenario_prob)
            st.markdown(
                f"<div style='color:{col2};font-weight:500'>{risk_dot(col2)}{risk_label(scenario_prob)}</div>",
                unsafe_allow_html=True
            )

        with res3:
            cleared = (base_prob >= THRESHOLD) and (scenario_prob < THRESHOLD)
            if cleared:
                st.success(
                    f"This scenario brings the customer **below** the risk threshold ({THRESHOLD}).\n\n"
                    "The proposed retention action would de-risk this customer."
                )
            elif base_prob < THRESHOLD:
                st.info("This customer was already below the risk threshold in the base profile.")
            else:
                still_above = scenario_prob - THRESHOLD
                st.warning(
                    f"Customer is still **{still_above:.1%} above** the threshold.\n\n"
                    "Try reducing products to 2, or activating membership."
                )

    # ── SIDE-BY-SIDE WATERFALL ───────────────────
    st.subheader("Feature contribution comparison")
    st.caption("Left = base profile · Right = after scenario changes")

    explainer_wif = get_shap_explainer(model)
    sv_base     = explainer_wif.shap_values(base_df)
    sv_scenario = explainer_wif.shap_values(scenario_df)

    wf1, wf2 = st.columns(2)

    with wf1, st.container(border=True, height="stretch"):
        st.markdown("**Base profile**")
        shap.waterfall_plot(
            shap.Explanation(
                values      = sv_base[0],
                base_values = float(np.array(explainer_wif.expected_value).flat[0]),
                data        = base_df.iloc[0],
                feature_names=FEATURE_COLS,
            ),
            show=False
        )
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close("all")

    with wf2, st.container(border=True, height="stretch"):
        st.markdown("**After scenario adjustments**")
        shap.waterfall_plot(
            shap.Explanation(
                values      = sv_scenario[0],
                base_values = float(np.array(explainer_wif.expected_value).flat[0]),
                data        = scenario_df.iloc[0],
                feature_names=FEATURE_COLS,
            ),
            show=False
        )
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close("all")

    st.caption(
        "Comparing these two waterfall plots shows exactly which feature "
        "contributions changed — and by how much — as a result of the retention actions."
    )