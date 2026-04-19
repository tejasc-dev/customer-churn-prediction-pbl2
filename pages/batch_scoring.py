"""
Batch Scoring — upload a CSV of customers, get churn scores back.
This is the feature that makes the app useful to a real business.
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Batch Scoring | Churn Intelligence",
    page_icon="⚡",
    layout="wide",
)

# ── Paths & model download ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH= BASE_DIR / "scaler.pkl"

MODEL_URL = "https://drive.google.com/uc?id=1swATBL3laAMhf97nIZs9tZgfJz-TsqwA"

@st.cache_resource(show_spinner="Downloading model…")
def ensure_model():
    if not MODEL_PATH.exists():
        gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)
    return True

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    ensure_model()
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html,[class*="css"]{font-family:'DM Sans',system-ui,sans-serif}
.stApp{background:linear-gradient(165deg,#080c11 0%,#0f141c 42%,#0a0e14 100%)}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1100px;padding-top:1.5rem}
.metric-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
  border-radius:14px;padding:1.1rem 1.4rem;text-align:center}
.metric-val{font-size:2rem;font-weight:700;line-height:1.1;margin-bottom:.2rem}
.metric-lbl{font-size:.72rem;font-weight:600;text-transform:uppercase;
  letter-spacing:.1em;color:rgba(148,163,184,.9)}
.risk-high{color:#fb7185}.risk-med{color:#fbbf24}.risk-low{color:#4ade80}
.badge{display:inline-block;padding:.2rem .65rem;border-radius:999px;
  font-size:.72rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.badge-high{background:rgba(239,68,68,.18);color:#fca5a5;border:1px solid rgba(239,68,68,.3)}
.badge-med{background:rgba(234,179,8,.18);color:#fde047;border:1px solid rgba(234,179,8,.3)}
.badge-low{background:rgba(34,197,94,.18);color:#86efac;border:1px solid rgba(34,197,94,.3)}
.section-head{font-size:.7rem;font-weight:700;letter-spacing:.14em;
  text-transform:uppercase;color:rgba(148,163,184,.9);margin:1.5rem 0 .6rem}
.upload-box{border:2px dashed rgba(99,102,241,.4);border-radius:16px;
  padding:2rem;text-align:center;background:rgba(99,102,241,.05)}
</style>
""", unsafe_allow_html=True)

# ── Feature engineering (mirrors training pipeline) ───────────────────────────
def tenure_band(t):
    if t <= 12: return 0
    if t <= 24: return 1
    if t <= 48: return 2
    return 3

def charge_ratio(total, monthly, tenure):
    denom = 7.0 * float(monthly) * max(int(tenure), 1) + 1.0
    return float(total) / denom

def avg_monthly_spend(total, tenure):
    return float(total) / float(tenure + 6)

def service_count(row):
    cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
            "TechSupport","StreamingTV","StreamingMovies"]
    return sum(1 for c in cols if str(row.get(c,"")).strip() == "Yes")

def high_risk_flag(row):
    if str(row.get("PaymentMethod","")) == "Electronic check":
        return 1
    if (str(row.get("InternetService","")) == "Fiber optic"
            and str(row.get("OnlineSecurity","")) == "No"
            and str(row.get("TechSupport","")) == "No"):
        return 1
    return 0

def ultra_loyal(row):
    t = int(row.get("tenure", 0))
    c = str(row.get("Contract",""))
    return 1 if (t >= 48 or c == "Two year") else 0

def engineer_row(row):
    t  = float(row.get("tenure", 0))
    mc = float(row.get("MonthlyCharges", 0))
    tc = float(row.get("TotalCharges", 0))
    return {
        "SeniorCitizen": float(row.get("SeniorCitizen", 0)),
        "tenure": t,
        "MonthlyCharges": mc,
        "TotalCharges": tc,
        "ChargeRatio": charge_ratio(tc, mc, t),
        "AvgMonthlySpend": avg_monthly_spend(tc, t),
        "TenureBand": float(tenure_band(int(t))),
        "ServiceCount": float(service_count(row)),
        "HighRiskFlag": float(high_risk_flag(row)),
        "UltraLoyal": float(ultra_loyal(row)),
        "gender_Male": 1.0 if str(row.get("gender","")) == "Male" else 0.0,
        "Partner_Yes": 1.0 if str(row.get("Partner","")) == "Yes" else 0.0,
        "Dependents_Yes": 1.0 if str(row.get("Dependents","")) == "Yes" else 0.0,
        "PhoneService_Yes": 1.0 if str(row.get("PhoneService","")) == "Yes" else 0.0,
        "MultipleLines_No phone service": 1.0 if str(row.get("MultipleLines","")) == "No phone service" else 0.0,
        "MultipleLines_Yes": 1.0 if str(row.get("MultipleLines","")) == "Yes" else 0.0,
        "InternetService_Fiber optic": 1.0 if str(row.get("InternetService","")) == "Fiber optic" else 0.0,
        "InternetService_No": 1.0 if str(row.get("InternetService","")) == "No" else 0.0,
        "OnlineSecurity_No internet service": 1.0 if str(row.get("OnlineSecurity","")) == "No internet service" else 0.0,
        "OnlineSecurity_Yes": 1.0 if str(row.get("OnlineSecurity","")) == "Yes" else 0.0,
        "OnlineBackup_No internet service": 1.0 if str(row.get("OnlineBackup","")) == "No internet service" else 0.0,
        "OnlineBackup_Yes": 1.0 if str(row.get("OnlineBackup","")) == "Yes" else 0.0,
        "DeviceProtection_No internet service": 1.0 if str(row.get("DeviceProtection","")) == "No internet service" else 0.0,
        "DeviceProtection_Yes": 1.0 if str(row.get("DeviceProtection","")) == "Yes" else 0.0,
        "TechSupport_No internet service": 1.0 if str(row.get("TechSupport","")) == "No internet service" else 0.0,
        "TechSupport_Yes": 1.0 if str(row.get("TechSupport","")) == "Yes" else 0.0,
        "StreamingTV_No internet service": 1.0 if str(row.get("StreamingTV","")) == "No internet service" else 0.0,
        "StreamingTV_Yes": 1.0 if str(row.get("StreamingTV","")) == "Yes" else 0.0,
        "StreamingMovies_No internet service": 1.0 if str(row.get("StreamingMovies","")) == "No internet service" else 0.0,
        "StreamingMovies_Yes": 1.0 if str(row.get("StreamingMovies","")) == "Yes" else 0.0,
        "Contract_One year": 1.0 if str(row.get("Contract","")) == "One year" else 0.0,
        "Contract_Two year": 1.0 if str(row.get("Contract","")) == "Two year" else 0.0,
        "PaperlessBilling_Yes": 1.0 if str(row.get("PaperlessBilling","")) == "Yes" else 0.0,
        "PaymentMethod_Credit card (automatic)": 1.0 if str(row.get("PaymentMethod","")) == "Credit card (automatic)" else 0.0,
        "PaymentMethod_Electronic check": 1.0 if str(row.get("PaymentMethod","")) == "Electronic check" else 0.0,
        "PaymentMethod_Mailed check": 1.0 if str(row.get("PaymentMethod","")) == "Mailed check" else 0.0,
    }

def score_dataframe(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    feature_names = list(scaler.feature_names_in_)
    rows = []
    for _, row in df.iterrows():
        engineered = engineer_row(row)
        vec = np.array([[engineered.get(f, 0.0) for f in feature_names]])
        rows.append(vec[0])
    X = np.array(rows)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, list(model.classes_).index(1)]
    result = df.copy()
    result["churn_probability"] = proba.round(4)
    result["churn_probability_pct"] = (proba * 100).round(1)
    result["risk_tier"] = pd.cut(
        proba,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    result["prediction"] = (proba >= 0.38).astype(int).map({0: "Will Stay", 1: "Will Churn"})
    return result.sort_values("churn_probability", ascending=False).reset_index(drop=True)

# ── Template CSV ──────────────────────────────────────────────────────────────
TEMPLATE_COLS = [
    "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService","OnlineSecurity",
    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
    "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges",
]
TEMPLATE_ROWS = [
    ["CUST-001","Female",0,"Yes","No",24,"Yes","No","Fiber optic","No","Yes","No","No","No","No","Month-to-month","Yes","Electronic check",89.10,2140.4],
    ["CUST-002","Male",0,"No","No",72,"Yes","Yes","DSL","Yes","Yes","Yes","Yes","No","No","Two year","No","Bank transfer (automatic)",56.95,4086.0],
    ["CUST-003","Female",1,"Yes","No",3,"No","No phone service","DSL","No","No","No","No","No","No","Month-to-month","Yes","Mailed check",34.20,102.6],
]

def make_template_csv() -> bytes:
    df = pd.DataFrame(TEMPLATE_ROWS, columns=TEMPLATE_COLS)
    return df.to_csv(index=False).encode()

# ── Risk tier badge HTML ───────────────────────────────────────────────────────
def badge_html(tier: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-med", "Low": "badge-low"}.get(tier, "badge-low")
    return f'<span class="badge {cls}">{tier}</span>'

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
  <p style="font-size:.7rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;
     color:rgba(129,140,248,.88);margin:0 0 .3rem">Churn Intelligence · Batch Mode</p>
  <h1 style="font-size:1.9rem;font-weight:700;letter-spacing:-.04em;margin:0 0 .4rem;
     background:linear-gradient(115deg,#f8fafc,#c7d2fe 38%,#38bdf8);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    Batch Customer Scoring
  </h1>
  <p style="color:rgba(148,163,184,.9);font-size:.95rem;margin:0">
    Upload your customer list — get a ranked risk report back in seconds.
  </p>
</div>
<hr style="border:0;height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,.5),rgba(56,189,248,.5),transparent);margin-bottom:1.5rem">
""", unsafe_allow_html=True)

# Load model
try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ── Step 1: Template download ─────────────────────────────────────────────────
st.markdown('<p class="section-head">Step 1 — Download the template</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    <div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);
         border-radius:12px;padding:1rem 1.2rem;font-size:.88rem;color:rgba(203,213,225,.95)">
      Your CSV must have these exact column names. Download the template, fill it with your
      customer data, then upload below. The <code>customerID</code> column is optional but
      useful for identifying customers in the results.
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.download_button(
        label="Download template CSV",
        data=make_template_csv(),
        file_name="churn_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown('<p class="section-head" style="margin-top:1.5rem">Step 2 — Upload your customer data</p>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"],
    help="Max ~50,000 rows. Must contain the columns in the template.",
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div class="upload-box">
      <p style="font-size:1rem;font-weight:500;color:#e2e8f0;margin:0 0 .4rem">
        Drop your CSV here or click Browse
      </p>
      <p style="font-size:.82rem;color:rgba(148,163,184,.85);margin:0">
        Required columns: tenure · MonthlyCharges · TotalCharges · Contract · InternetService and others
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Parse and validate ────────────────────────────────────────────────────────
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

required = {"tenure", "MonthlyCharges", "TotalCharges", "Contract",
            "InternetService", "PhoneService"}
missing_cols = required - set(df_raw.columns)
if missing_cols:
    st.error(f"Missing required columns: {', '.join(sorted(missing_cols))}. "
             f"Download the template above and check your column names.")
    st.stop()

# Coerce numeric
for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)

n_rows = len(df_raw)
st.success(f"Loaded {n_rows:,} customers. Running churn scores…")

# ── Score ─────────────────────────────────────────────────────────────────────
with st.spinner("Scoring…"):
    df_scored = score_dataframe(df_raw, model, scaler)

# ── Summary metrics ───────────────────────────────────────────────────────────
n_high   = int((df_scored["risk_tier"] == "High").sum())
n_med    = int((df_scored["risk_tier"] == "Medium").sum())
n_low    = int((df_scored["risk_tier"] == "Low").sum())
avg_prob = float(df_scored["churn_probability"].mean())

st.markdown('<p class="section-head" style="margin-top:1.5rem">Step 3 — Review results</p>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val risk-high">{n_high}</div>
      <div class="metric-lbl">High risk</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val risk-med">{n_med}</div>
      <div class="metric-lbl">Medium risk</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val risk-low">{n_low}</div>
      <div class="metric-lbl">Low risk</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="color:#c7d2fe">{avg_prob*100:.1f}%</div>
      <div class="metric-lbl">Avg churn prob</div>
    </div>""", unsafe_allow_html=True)

# ── Charts row ─────────────────────────────────────────────────────────────────
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_pie = go.Figure(go.Pie(
        labels=["High risk", "Medium risk", "Low risk"],
        values=[n_high, n_med, n_low],
        hole=0.55,
        marker_colors=["#ef4444", "#eab308", "#22c55e"],
        textinfo="percent+label",
        textfont=dict(size=12),
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1", showlegend=False,
        margin=dict(t=20, b=10, l=10, r=10), height=240,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    fig_hist = px.histogram(
        df_scored, x="churn_probability_pct",
        nbins=20, color_discrete_sequence=["#6366f1"],
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1", xaxis_title="Churn probability (%)",
        yaxis_title="Customers", margin=dict(t=20, b=40, l=40, r=10),
        height=240, bargap=0.05,
    )
    fig_hist.update_xaxes(showgrid=False)
    fig_hist.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Scored table ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Ranked customer list (highest risk first)</p>', unsafe_allow_html=True)

# Build display dataframe
display_cols = []
if "customerID" in df_scored.columns:
    display_cols.append("customerID")
display_cols += ["tenure", "Contract", "MonthlyCharges",
                 "churn_probability_pct", "risk_tier", "prediction"]
display_cols = [c for c in display_cols if c in df_scored.columns]

df_display = df_scored[display_cols].copy()
df_display = df_display.rename(columns={
    "churn_probability_pct": "Churn prob %",
    "risk_tier": "Risk tier",
    "prediction": "Prediction",
    "customerID": "Customer ID",
    "tenure": "Tenure (mo)",
    "MonthlyCharges": "Monthly charge",
})

# Colour-map the Risk tier column
def colour_risk(val):
    if val == "High":   return "background-color:rgba(239,68,68,.15);color:#fca5a5"
    if val == "Medium": return "background-color:rgba(234,179,8,.15);color:#fde047"
    return "background-color:rgba(34,197,94,.15);color:#86efac"

def colour_pred(val):
    if val == "Will Churn": return "color:#fb7185;font-weight:600"
    return "color:#4ade80;font-weight:600"

styled = (
    df_display.style
    .applymap(colour_risk, subset=["Risk tier"] if "Risk tier" in df_display.columns else [])
    .applymap(colour_pred, subset=["Prediction"] if "Prediction" in df_display.columns else [])
    .format({"Churn prob %": "{:.1f}%", "Monthly charge": "₹{:.0f}"})
    .set_properties(**{"font-size": "13px"})
)

st.dataframe(styled, use_container_width=True, height=420)

# ── Download ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Step 4 — Download your scored report</p>', unsafe_allow_html=True)

dl_df = df_scored.copy()
dl_df = dl_df.rename(columns={"churn_probability_pct": "Churn_Probability_Pct",
                                "risk_tier": "Risk_Tier",
                                "prediction": "Prediction"})

col_dl1, col_dl2, col_dl3 = st.columns(3)

with col_dl1:
    st.download_button(
        label="Download full report (CSV)",
        data=dl_df.to_csv(index=False).encode(),
        file_name="churn_scores_full.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    high_risk_df = dl_df[dl_df["Risk_Tier"] == "High"]
    st.download_button(
        label=f"Download high-risk only ({n_high} customers)",
        data=high_risk_df.to_csv(index=False).encode(),
        file_name="churn_scores_high_risk.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )

with col_dl3:
    # Action plan: only key columns + recommendation
    action_df = dl_df[
        ([c for c in ["customerID","tenure","Contract","MonthlyCharges",
                       "Churn_Probability_Pct","Risk_Tier","Prediction"]
          if c in dl_df.columns])
    ].copy()
    action_df["Recommended_Action"] = action_df["Risk_Tier"].map({
        "High":   "Priority outreach — offer discount or upgrade within 7 days",
        "Medium": "Send proactive check-in email or loyalty offer",
        "Low":    "No immediate action required",
    })
    st.download_button(
        label="Download action plan (CSV)",
        data=action_df.to_csv(index=False).encode(),
        file_name="churn_action_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Contact CTA ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.25);
     border-radius:16px;padding:1.4rem 1.6rem;display:flex;justify-content:space-between;
     align-items:center;flex-wrap:wrap;gap:1rem">
  <div>
    <p style="font-size:1rem;font-weight:600;color:#e2e8f0;margin:0 0 .3rem">
      Want this running on your own customer data every month?
    </p>
    <p style="font-size:.85rem;color:rgba(148,163,184,.9);margin:0">
      Get a custom deployment, automatic monthly reports, and a private dashboard.
    </p>
  </div>
  <a href="https://wa.me/919389860636?text=Hi%20Tejas%2C%20I%20want%20churn%20prediction%20for%20my%20business"
     target="_blank"
     style="background:linear-gradient(135deg,#4f46e5,#0891b2);color:#fff;
            font-size:.85rem;font-weight:700;padding:.65rem 1.3rem;border-radius:10px;
            text-decoration:none;white-space:nowrap">
    Message on WhatsApp
  </a>
</div>
""", unsafe_allow_html=True)