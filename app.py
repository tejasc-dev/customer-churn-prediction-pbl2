"""
Production-quality Streamlit application for customer churn prediction.

Loads a trained classifier and StandardScaler, applies the same preprocessing
as training, and surfaces predictions with risk tiers and visual diagnostics.
"""


from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1swATBL3laAMhf97nIZs9tZgfJz-TsqwA"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

APP_TITLE = "Churn Intelligence"
APP_TAGLINE = "Subscriber retention scoring for network operations"
# Telecom-style product branding (customize for your organization)
BRAND_COMPANY = "NorthStar Telecom"
BRAND_PRODUCT_LINE = "Retention Command Center"
BRAND_BADGE = "Live scoring"
FOOTER_LEGAL = "Confidential — authorized revenue and retention use only."
PROJECT_VERSION = "1.0.0"

RISK_LOW_MAX = 0.4
RISK_HIGH_MIN = 0.7


# ---------------------------------------------------------------------------
# Theme: global CSS (dark, minimal, glass-style surfaces)
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    """Telecom analytics dashboard theme: spacing scale, motion, glass surfaces."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

            :root {
                --space-xs: 0.35rem;
                --space-sm: 0.65rem;
                --space-md: 1rem;
                --space-lg: 1.5rem;
                --space-xl: 2rem;
                --radius-sm: 10px;
                --radius-md: 14px;
                --radius-lg: 18px;
                --text-muted: rgba(148, 163, 184, 0.92);
                --text-body: #e8edf4;
                --border-subtle: rgba(255, 255, 255, 0.08);
            }

            html, body, [class*="css"] {
                font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
                color: var(--text-body);
            }

            .stApp {
                background:
                    radial-gradient(1000px 520px at 0% -5%, rgba(30, 58, 95, 0.55) 0%, transparent 50%),
                    radial-gradient(800px 480px at 100% 0%, rgba(49, 46, 129, 0.45) 0%, transparent 48%),
                    linear-gradient(165deg, #080c11 0%, #0f141c 42%, #0a0e14 100%);
                color: var(--text-body);
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header[data-testid="stHeader"] {
                background: rgba(8, 12, 18, 0.9);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid var(--border-subtle);
            }

            .block-container {
                max-width: 1080px;
                padding-top: var(--space-lg);
                padding-bottom: 6.5rem;
                padding-left: var(--space-md) !important;
                padding-right: var(--space-md) !important;
            }

            @keyframes resultEntryAnim {
                from { opacity: 0; transform: translateY(16px) scale(0.985); }
                to { opacity: 1; transform: translateY(0) scale(1); }
            }
            @keyframes recActionAnim {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .result-entry-anim {
                animation: resultEntryAnim 0.58s cubic-bezier(0.22, 1, 0.36, 1) both;
            }
            .rec-action-anim {
                animation: recActionAnim 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.08s both;
            }

            /* Command header */
            .dash-header-wrap {
                margin: 0 0 var(--space-lg) 0;
                padding: 0 0.15rem;
            }
            .dash-header {
                display: flex;
                flex-direction: row;
                align-items: stretch;
                gap: 0;
                padding: var(--space-sm) 0 var(--space-md) 0;
            }
            .dash-header-logo {
                flex-shrink: 0;
                width: 56px;
                height: 56px;
                align-self: center;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: var(--radius-md);
                background: linear-gradient(145deg, rgba(99, 102, 241, 0.28), rgba(14, 165, 233, 0.12));
                border: 1px solid rgba(255, 255, 255, 0.14);
                box-shadow:
                    0 0 0 1px rgba(0, 0, 0, 0.28) inset,
                    0 12px 36px rgba(79, 70, 229, 0.22);
            }
            .dash-header-logo-svg {
                display: block;
                width: 32px;
                height: 32px;
            }
            .dash-header-rail {
                width: 1px;
                align-self: stretch;
                margin: 0 1.1rem;
                background: linear-gradient(180deg, transparent, rgba(255,255,255,0.14), transparent);
                min-height: 56px;
            }
            .dash-header-body {
                flex: 1;
                min-width: 0;
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 0.35rem;
            }
            .dash-header-kicker {
                font-size: 0.68rem;
                font-weight: 600;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                color: rgba(129, 140, 248, 0.88);
                margin: 0;
            }
            .dash-header-title-row {
                display: flex;
                flex-wrap: wrap;
                align-items: baseline;
                gap: 0.65rem 1rem;
            }
            .dash-header-title {
                font-size: clamp(1.45rem, 2.8vw, 1.95rem);
                font-weight: 700;
                letter-spacing: -0.045em;
                line-height: 1.12;
                margin: 0;
                padding: 0;
                background: linear-gradient(115deg, #f8fafc 0%, #c7d2fe 38%, #38bdf8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .dash-header-badge {
                font-size: 0.68rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #a5f3fc;
                background: rgba(6, 182, 212, 0.12);
                border: 1px solid rgba(34, 211, 238, 0.28);
                border-radius: 999px;
                padding: 0.2rem 0.55rem;
            }
            .dash-header-tagline {
                margin: 0;
                padding: 0;
                font-size: 0.92rem;
                font-weight: 500;
                color: var(--text-muted);
                letter-spacing: 0.02em;
                line-height: 1.5;
                max-width: 38rem;
            }
            .dash-header-divider {
                height: 1px;
                width: 100%;
                margin: 0;
                border: 0;
                border-radius: 1px;
                background: linear-gradient(
                    90deg,
                    transparent 0%,
                    rgba(99, 102, 241, 0.2) 10%,
                    rgba(99, 102, 241, 0.55) 30%,
                    rgba(56, 189, 248, 0.5) 50%,
                    rgba(129, 140, 248, 0.52) 72%,
                    rgba(99, 102, 241, 0.15) 90%,
                    transparent 100%
                );
                box-shadow: 0 1px 16px rgba(56, 189, 248, 0.16);
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(195deg, rgba(22, 28, 38, 0.98) 0%, rgba(10, 14, 20, 0.99) 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.07);
            }
            [data-testid="stSidebar"] .block-container {
                padding-top: var(--space-md);
            }
            .sidebar-brand {
                padding: 0.15rem 0 1rem 0;
                margin-bottom: 0.25rem;
                border-bottom: 1px solid rgba(255,255,255,0.06);
            }
            .sidebar-brand-name {
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: #e2e8f0;
                margin: 0 0 0.15rem 0;
            }
            .sidebar-brand-product {
                font-size: 0.78rem;
                color: rgba(148, 163, 184, 0.95);
                margin: 0;
                line-height: 1.35;
            }
            .sidebar-hint {
                font-size: 0.78rem;
                line-height: 1.45;
                color: rgba(148, 163, 184, 0.9);
                background: rgba(99, 102, 241, 0.08);
                border: 1px solid rgba(99, 102, 241, 0.15);
                border-radius: var(--radius-sm);
                padding: 0.55rem 0.65rem;
                margin: 0 0 0.75rem 0;
            }
            .sidebar-section-head {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                margin: 1.1rem 0 0.45rem 0;
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: rgba(148, 163, 184, 0.95);
            }
            .sidebar-section-num {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 1.35rem;
                height: 1.35rem;
                border-radius: 6px;
                font-size: 0.65rem;
                font-weight: 800;
                color: #c7d2fe;
                background: rgba(99, 102, 241, 0.2);
                border: 1px solid rgba(129, 140, 248, 0.35);
            }

            .section-label {
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: rgba(148, 163, 184, 0.95);
                margin: 1.25rem 0 0.5rem 0;
                font-weight: 600;
            }
            hr.soft {
                border: none;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                margin: 1.15rem 0;
            }

            /* Result hero (HTML block) */
            .result-hero-wrap {
                border-radius: var(--radius-lg);
                padding: var(--space-lg) var(--space-xl);
                margin-bottom: var(--space-lg);
                background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
            }
            .result-hero-grid {
                display: grid;
                grid-template-columns: 1fr auto;
                gap: var(--space-lg);
                align-items: center;
            }
            @media (max-width: 720px) {
                .result-hero-grid { grid-template-columns: 1fr; }
            }
            .result-hero-label {
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: rgba(148, 163, 184, 0.88);
                margin-bottom: var(--space-xs);
            }
            .result-hero-churn {
                font-size: clamp(2.75rem, 8vw, 4.25rem);
                font-weight: 800;
                letter-spacing: -0.04em;
                line-height: 1;
                margin: 0 0 var(--space-sm) 0;
            }
            .result-hero-churn-yes { color: #fb7185; text-shadow: 0 0 40px rgba(251, 113, 133, 0.25); }
            .result-hero-churn-no { color: #4ade80; text-shadow: 0 0 40px rgba(74, 222, 128, 0.2); }
            .result-hero-sub {
                font-size: 1.05rem;
                font-weight: 500;
                color: var(--text-muted);
                margin: 0;
            }
            .result-hero-prob-block {
                text-align: right;
            }
            .result-hero-prob-val {
                font-size: clamp(2rem, 5vw, 2.75rem);
                font-weight: 700;
                letter-spacing: -0.03em;
                color: #f1f5f9;
                line-height: 1.1;
            }
            .result-hero-prob-cap {
                font-size: 0.8rem;
                color: rgba(148, 163, 184, 0.9);
                margin-top: 0.35rem;
            }
            .result-hero-pills {
                margin-top: var(--space-md);
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                align-items: center;
            }

            .recommended-panel {
                border-radius: var(--radius-md);
                padding: var(--space-md) var(--space-lg);
                margin-bottom: var(--space-lg);
                border: 1px solid rgba(255,255,255,0.1);
                background: rgba(15, 23, 42, 0.45);
            }
            .recommended-panel h4 {
                margin: 0 0 0.5rem 0;
                font-size: 0.95rem;
                font-weight: 700;
                color: #f1f5f9;
            }
            .recommended-panel p {
                margin: 0;
                font-size: 0.88rem;
                line-height: 1.55;
                color: rgba(203, 213, 225, 0.95);
            }
            .recommended-panel.low { border-left: 4px solid #22c55e; }
            .recommended-panel.med { border-left: 4px solid #eab308; }
            .recommended-panel.high { border-left: 4px solid #ef4444; }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                border-radius: var(--radius-lg) !important;
                padding: var(--space-sm) var(--space-md) var(--space-md) var(--space-md);
                backdrop-filter: blur(18px);
                box-shadow: 0 22px 56px rgba(0, 0, 0, 0.4);
            }

            .metric-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.38rem 0.85rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                border: 1px solid rgba(255,255,255,0.12);
            }
            .pill-low { background: rgba(34, 197, 94, 0.18); color: #86efac; }
            .pill-med { background: rgba(234, 179, 8, 0.18); color: #fde047; }
            .pill-high { background: rgba(239, 68, 68, 0.18); color: #fca5a5; }

            .dash-charts-title {
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: rgba(148, 163, 184, 0.9);
                margin: 0.5rem 0 0.35rem 0;
            }

            .footer-bar {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                z-index: 999;
                padding: 0.55rem 1.25rem 0.65rem;
                background: rgba(6, 9, 14, 0.92);
                border-top: 1px solid rgba(255, 255, 255, 0.07);
                backdrop-filter: blur(14px);
                box-shadow: 0 -8px 32px rgba(0, 0, 0, 0.35);
            }
            .footer-bar-inner {
                max-width: 1080px;
                margin: 0 auto;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.2rem;
                text-align: center;
            }
            .footer-brand {
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                color: #cbd5e1;
            }
            .footer-meta {
                font-size: 0.74rem;
                color: rgba(148, 163, 184, 0.88);
            }
            .footer-legal {
                font-size: 0.68rem;
                color: rgba(100, 116, 139, 0.95);
                max-width: 52rem;
                line-height: 1.4;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionResult:
    """Structured output from the classifier."""

    churn: bool
    churn_label: str
    probability_churn: float
    probability_retain: float
    risk_level: str
    risk_css_class: str


# ---------------------------------------------------------------------------
# Feature engineering (matches training-time engineered columns)
# ---------------------------------------------------------------------------


def _tenure_band(tenure: int) -> int:
    if tenure <= 12:
        return 0
    if tenure <= 24:
        return 1
    if tenure <= 48:
        return 2
    return 3


def _compute_charge_ratio(total_charges: float, monthly_charges: float, tenure: int) -> float:
    t = max(int(tenure), 1)
    denom = 7.0 * float(monthly_charges) * float(t) + 1.0
    return float(total_charges) / denom


def _compute_avg_monthly_spend(total_charges: float, tenure: int) -> float:
    return float(total_charges) / float(tenure + 6)


def _service_count(
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
) -> int:
    return sum(
        1
        for s in (
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        )
        if s == "Yes"
    )


def _high_risk_flag(
    payment_method: str,
    internet_service: str,
    online_security: str,
    tech_support: str,
) -> int:
    if payment_method == "Electronic check":
        return 1
    if (
        internet_service == "Fiber optic"
        and online_security == "No"
        and tech_support == "No"
    ):
        return 1
    return 0


def _ultra_loyal(tenure: int, contract: str) -> int:
    return 1 if (tenure >= 48 or contract == "Two year") else 0


def _build_feature_row(
    *,
    senior_citizen: int,
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    gender: str,
    partner: str,
    dependents: str,
    phone_service: str,
    multiple_lines: str,
    internet_service: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    contract: str,
    paperless_billing: str,
    payment_method: str,
) -> dict[str, float]:
    """Map raw UI fields to the exact numeric / one-hot vector expected by the scaler."""
    cr = _compute_charge_ratio(total_charges, monthly_charges, tenure)
    ams = _compute_avg_monthly_spend(total_charges, tenure)
    tb = float(_tenure_band(tenure))
    sc = float(
        _service_count(
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        )
    )
    hrf = float(
        _high_risk_flag(
            payment_method,
            internet_service,
            online_security,
            tech_support,
        )
    )
    ul = float(_ultra_loyal(tenure, contract))

    return {
        "SeniorCitizen": float(senior_citizen),
        "tenure": float(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "ChargeRatio": cr,
        "AvgMonthlySpend": ams,
        "TenureBand": tb,
        "ServiceCount": sc,
        "HighRiskFlag": hrf,
        "UltraLoyal": ul,
        "gender_Male": 1.0 if gender == "Male" else 0.0,
        "Partner_Yes": 1.0 if partner == "Yes" else 0.0,
        "Dependents_Yes": 1.0 if dependents == "Yes" else 0.0,
        "PhoneService_Yes": 1.0 if phone_service == "Yes" else 0.0,
        "MultipleLines_No phone service": 1.0 if multiple_lines == "No phone service" else 0.0,
        "MultipleLines_Yes": 1.0 if multiple_lines == "Yes" else 0.0,
        "InternetService_Fiber optic": 1.0 if internet_service == "Fiber optic" else 0.0,
        "InternetService_No": 1.0 if internet_service == "No" else 0.0,
        "OnlineSecurity_No internet service": 1.0 if online_security == "No internet service" else 0.0,
        "OnlineSecurity_Yes": 1.0 if online_security == "Yes" else 0.0,
        "OnlineBackup_No internet service": 1.0 if online_backup == "No internet service" else 0.0,
        "OnlineBackup_Yes": 1.0 if online_backup == "Yes" else 0.0,
        "DeviceProtection_No internet service": 1.0
        if device_protection == "No internet service"
        else 0.0,
        "DeviceProtection_Yes": 1.0 if device_protection == "Yes" else 0.0,
        "TechSupport_No internet service": 1.0 if tech_support == "No internet service" else 0.0,
        "TechSupport_Yes": 1.0 if tech_support == "Yes" else 0.0,
        "StreamingTV_No internet service": 1.0 if streaming_tv == "No internet service" else 0.0,
        "StreamingTV_Yes": 1.0 if streaming_tv == "Yes" else 0.0,
        "StreamingMovies_No internet service": 1.0
        if streaming_movies == "No internet service"
        else 0.0,
        "StreamingMovies_Yes": 1.0 if streaming_movies == "Yes" else 0.0,
        "Contract_One year": 1.0 if contract == "One year" else 0.0,
        "Contract_Two year": 1.0 if contract == "Two year" else 0.0,
        "PaperlessBilling_Yes": 1.0 if paperless_billing == "Yes" else 0.0,
        "PaymentMethod_Credit card (automatic)": 1.0
        if payment_method == "Credit card (automatic)"
        else 0.0,
        "PaymentMethod_Electronic check": 1.0 if payment_method == "Electronic check" else 0.0,
        "PaymentMethod_Mailed check": 1.0 if payment_method == "Mailed check" else 0.0,
    }


def _row_to_matrix(scaler: Any, row: dict[str, float]) -> np.ndarray:
    names = list(scaler.feature_names_in_)
    missing = set(names) - set(row.keys())
    if missing:
        raise ValueError(f"Missing engineered features: {sorted(missing)}")
    return np.array([[row[name] for name in names]], dtype=np.float64)


def _risk_tier(probability_churn: float) -> tuple[str, str]:
    """Return (label, css_class_suffix) for risk styling."""
    if probability_churn < RISK_LOW_MAX:
        return "Low", "low"
    if probability_churn <= RISK_HIGH_MIN:
        return "Medium", "med"
    return "High", "high"


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """
    Load the serialized classifier from disk.

    Raises:
        FileNotFoundError: If model.pkl is missing.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_scaler() -> Any:
    """
    Load the StandardScaler fit on training features.

    Raises:
        FileNotFoundError: If scaler.pkl is missing.
    """
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    return joblib.load(SCALER_PATH)


def preprocess_input(raw_fields: dict[str, Any], scaler: Any) -> tuple[np.ndarray, dict[str, float]]:
    """
    Engineer features, align column order with the scaler, and return scaled features.

    Returns:
        X_scaled: shape (1, n_features) — ready for model.predict / predict_proba.
        feature_row: raw engineered dict for diagnostics / transparency.
    """
    row = _build_feature_row(**raw_fields)
    X_raw = _row_to_matrix(scaler, row)
    X_scaled = scaler.transform(X_raw)
    return X_scaled, row


def predict(model: Any, X_scaled: np.ndarray) -> PredictionResult:
    """
    Run inference and package outputs for the UI.

    Assumes positive class label == 1 represents churn.
    """
    proba = model.predict_proba(X_scaled)[0]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model classes must include label 1 for churn.")
    pos_idx = classes.index(1)
    churn_prob = float(proba[pos_idx])
    retain_prob = float(1.0 - churn_prob)
    pred = int(model.predict(X_scaled)[0])
    churn = pred == 1
    risk_label, risk_class = _risk_tier(churn_prob)
    return PredictionResult(
        churn=churn,
        churn_label="Yes" if churn else "No",
        probability_churn=churn_prob,
        probability_retain=retain_prob,
        risk_level=risk_label,
        risk_css_class=risk_class,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_inputs(
    *,
    phone_service: str,
    multiple_lines: str,
    internet_service: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    monthly_charges: float,
    total_charges: float,
) -> list[str]:
    """Return a list of user-facing validation messages (empty if valid)."""
    errors: list[str] = []
    if phone_service == "No" and multiple_lines != "No phone service":
        errors.append(
            "When **Phone service** is **No**, **Multiple lines** must be **No phone service**."
        )
    if internet_service == "No":
        for label, val in (
            ("Online security", online_security),
            ("Online backup", online_backup),
            ("Device protection", device_protection),
            ("Tech support", tech_support),
            ("Streaming TV", streaming_tv),
            ("Streaming movies", streaming_movies),
        ):
            if val != "No internet service":
                errors.append(
                    f"When **Internet service** is **No**, **{label}** must be **No internet service**."
                )
    if monthly_charges < 0 or total_charges < 0:
        errors.append("Charges cannot be negative.")
    if np.isnan(monthly_charges) or np.isnan(total_charges):
        errors.append("Charges must be valid numbers.")
    return errors


# ---------------------------------------------------------------------------
# Session state & defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 12,
    "contract": "Month-to-month",
    "paperless_billing": "No",
    "payment_method": "Electronic check",
    "monthly_charges": 65.0,
    "total_charges": 800.0,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
}


def _init_session_state() -> None:
    """Seed widget-backed session keys once per browser session."""
    for key, val in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "has_prediction" not in st.session_state:
        st.session_state["has_prediction"] = False


def _reset_inputs() -> None:
    """Restore defaults and clear the last prediction."""
    for key, val in _DEFAULTS.items():
        st.session_state[key] = val
    st.session_state["has_prediction"] = False
    st.session_state.pop("prediction", None)
    st.session_state.pop("last_feature_row", None)


def _collect_raw_fields_from_state() -> dict[str, Any]:
    """Build kwargs for _build_feature_row from session_state."""
    return {
        "senior_citizen": int(st.session_state["senior_citizen"]),
        "tenure": int(st.session_state["tenure"]),
        "monthly_charges": float(st.session_state["monthly_charges"]),
        "total_charges": float(st.session_state["total_charges"]),
        "gender": st.session_state["gender"],
        "partner": st.session_state["partner"],
        "dependents": st.session_state["dependents"],
        "phone_service": st.session_state["phone_service"],
        "multiple_lines": st.session_state["multiple_lines"],
        "internet_service": st.session_state["internet_service"],
        "online_security": st.session_state["online_security"],
        "online_backup": st.session_state["online_backup"],
        "device_protection": st.session_state["device_protection"],
        "tech_support": st.session_state["tech_support"],
        "streaming_tv": st.session_state["streaming_tv"],
        "streaming_movies": st.session_state["streaming_movies"],
        "contract": st.session_state["contract"],
        "paperless_billing": st.session_state["paperless_billing"],
        "payment_method": st.session_state["payment_method"],
    }


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------


def _render_header() -> None:
    """Command-center style banner: mark, rail, hierarchy, product badge, rule."""
    st.markdown(
        f"""
        <div class="dash-header-wrap">
            <header class="dash-header" role="banner">
                <div class="dash-header-logo" title="{BRAND_COMPANY} · {APP_TITLE}">
                    <svg class="dash-header-logo-svg" viewBox="0 0 32 32" width="32" height="32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <rect width="32" height="32" rx="9" fill="url(#churnLogoGrad)"/>
                        <path d="M7.5 23.5V15M12.5 23.5V11M17.5 23.5V17M22.5 23.5V13" stroke="rgba(248,250,252,0.92)" stroke-width="2" stroke-linecap="round"/>
                        <defs>
                            <linearGradient id="churnLogoGrad" x1="6" y1="4" x2="26" y2="28" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#4f46e5"/>
                                <stop offset="0.5" stop-color="#7c3aed"/>
                                <stop offset="1" stop-color="#0891b2"/>
                            </linearGradient>
                        </defs>
                    </svg>
                </div>
                <div class="dash-header-rail" aria-hidden="true"></div>
                <div class="dash-header-body">
                    <p class="dash-header-kicker">{BRAND_COMPANY} · {BRAND_PRODUCT_LINE}</p>
                    <div class="dash-header-title-row">
                        <h1 class="dash-header-title">{APP_TITLE}</h1>
                        <span class="dash-header-badge">{BRAND_BADGE}</span>
                    </div>
                    <p class="dash-header-tagline">{APP_TAGLINE}</p>
                </div>
            </header>
            <div class="dash-header-divider" aria-hidden="true"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar_inputs() -> tuple[bool, bool]:
    """
    Render sidebar controls. Returns (predict_clicked, reset_clicked).
    """
    # Keep dependent fields consistent with product rules before widgets render.
    if st.session_state.get("phone_service") == "No":
        st.session_state["multiple_lines"] = "No phone service"
    if st.session_state.get("internet_service") == "No":
        for k in (
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ):
            st.session_state[k] = "No internet service"

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <p class="sidebar-brand-name">{BRAND_COMPANY}</p>
            <p class="sidebar-brand-product">{BRAND_PRODUCT_LINE}<br/><span style="opacity:0.85">Subscriber scoring workspace</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="sidebar-hint">
            <strong>Tip:</strong> match CRM fields to this panel, then run scoring. Phone and internet rules are enforced automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="sidebar-section-head"><span class="sidebar-section-num">1</span> Subscriber profile</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.selectbox(
        "Gender",
        ["Female", "Male"],
        key="gender",
        help="Encoded as in training (e.g. gender_Male).",
    )
    st.sidebar.selectbox(
        "Senior citizen",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        key="senior_citizen",
        help="SeniorCitizen flag from billing / CRM.",
    )
    st.sidebar.selectbox(
        "Partner on account",
        ["No", "Yes"],
        key="partner",
        help="Household partner indicator.",
    )
    st.sidebar.selectbox(
        "Dependents",
        ["No", "Yes"],
        key="dependents",
        help="Dependents indicator for family-plan context.",
    )
    st.sidebar.slider(
        "Tenure (months)",
        min_value=0,
        max_value=120,
        key="tenure",
        help="Months active on network; drives tenure bands and spend ratios.",
    )

    st.sidebar.markdown(
        '<div class="sidebar-section-head"><span class="sidebar-section-num">2</span> Network & products</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.selectbox(
        "Phone service",
        ["No", "Yes"],
        key="phone_service",
        help="If **No**, multiple lines must read **No phone service**.",
    )
    ml_options = (
        ["No phone service"]
        if st.session_state["phone_service"] == "No"
        else ["No phone service", "No", "Yes"]
    )
    st.sidebar.selectbox(
        "Multiple lines",
        ml_options,
        key="multiple_lines",
        help="Requires active phone; single-line vs multi-line.",
    )
    st.sidebar.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "No"],
        key="internet_service",
        help="If **No**, all add-ons switch to **No internet service**.",
    )

    addon_options = (
        ["No internet service"]
        if st.session_state["internet_service"] == "No"
        else ["No", "Yes"]
    )
    st.sidebar.caption("Add-ons (voice / broadband attach)")
    st.sidebar.selectbox(
        "Online security",
        addon_options,
        key="online_security",
        help="Security product; risk model uses with fiber context.",
    )
    st.sidebar.selectbox(
        "Online backup",
        addon_options,
        key="online_backup",
        help="Backup attach rate.",
    )
    st.sidebar.selectbox(
        "Device protection",
        addon_options,
        key="device_protection",
        help="Handset / CPE protection.",
    )
    st.sidebar.selectbox(
        "Tech support",
        addon_options,
        key="tech_support",
        help="Premium support; pairs with security for high-risk flags.",
    )
    st.sidebar.selectbox(
        "Streaming TV",
        addon_options,
        key="streaming_tv",
        help="IPTV / OTT TV attach.",
    )
    st.sidebar.selectbox(
        "Streaming movies",
        addon_options,
        key="streaming_movies",
        help="Movie streaming attach.",
    )

    st.sidebar.markdown(
        '<div class="sidebar-section-head"><span class="sidebar-section-num">3</span> Revenue & commitment</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
        key="contract",
        help="Commitment term; influences loyalty features.",
    )
    st.sidebar.selectbox(
        "Paperless billing",
        ["No", "Yes"],
        key="paperless_billing",
        help="Digital bill delivery preference.",
    )
    st.sidebar.selectbox(
        "Payment method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
        key="payment_method",
        help="Electronic check often correlates with higher churn in telco cohorts.",
    )
    st.sidebar.slider(
        "Monthly recurring charge ($)",
        min_value=0.0,
        max_value=200.0,
        step=0.5,
        key="monthly_charges",
        help="MRC from billing — used with tenure for engineered ratios.",
    )
    st.sidebar.number_input(
        "Lifetime revenue ($)",
        min_value=0.0,
        max_value=20000.0,
        step=1.0,
        key="total_charges",
        help="Cumulative billed revenue (align with finance / data warehouse).",
    )

    st.sidebar.markdown('<hr class="soft"/>', unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        predict_clicked = st.button(
            "Run scoring",
            type="primary",
            use_container_width=True,
            help="Execute model pipeline with current inputs.",
        )
    with c2:
        reset_clicked = st.button(
            "Reset",
            use_container_width=True,
            help="Restore defaults and clear the last scorecard.",
        )

    with st.sidebar.expander("Scoring methodology", expanded=False):
        st.markdown(
            f"""
            - **Artifacts:** `model.pkl` + `scaler.pkl`
            - **Transform:** engineered features → column order match → `StandardScaler`
            - **Target:** P(churn = **1**)
            - **Risk bands:** Low below {RISK_LOW_MAX:.0%} · Medium {RISK_LOW_MAX:.0%}–{RISK_HIGH_MIN:.0%} · High above {RISK_HIGH_MIN:.0%}
            """
        )

    return predict_clicked, reset_clicked


def _gauge_needle_color(probability_churn: float) -> str:
    """Accent for gauge bar and number from current churn probability."""
    p = probability_churn
    if p < RISK_LOW_MAX:
        return "#34d399"
    if p <= RISK_HIGH_MIN:
        return "#fbbf24"
    return "#fb7185"


def _gauge_figure(probability_churn: float) -> go.Figure:
    """Semi-circular gauge with band-colored track, dynamic needle, and labeled axis."""
    val_pct = float(np.clip(probability_churn * 100.0, 0.0, 100.0))
    needle = _gauge_needle_color(probability_churn)
    low_end = RISK_LOW_MAX * 100.0
    high_start = RISK_HIGH_MIN * 100.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val_pct,
            number={
                "suffix": "%",
                "font": {"size": 40, "color": needle, "family": "DM Sans, sans-serif"},
            },
            title={
                "text": (
                    "<b>Churn exposure</b><br>"
                    "<span style='font-size:11px;font-weight:500;color:#94a3b8'>"
                    "Green: low · Amber: elevated · Red: critical</span>"
                ),
                "font": {"size": 14, "color": "#e2e8f0", "family": "DM Sans, sans-serif"},
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "rgba(148,163,184,0.4)",
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0%", "25%", "50%", "75%", "100%"],
                    "tickfont": {"size": 10, "color": "#94a3b8"},
                },
                "bar": {"color": needle, "line": {"width": 0}, "thickness": 0.22},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, low_end], "color": "rgba(16,185,129,0.28)", "name": "Low"},
                    {"range": [low_end, high_start], "color": "rgba(245,158,11,0.26)", "name": "Medium"},
                    {"range": [high_start, 100], "color": "rgba(248,113,113,0.28)", "name": "High"},
                ],
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 2},
                    "thickness": 0.85,
                    "value": val_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=36, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#e8edf4"},
    )
    return fig


def _confidence_bar_figure(retain: float, churn: float) -> go.Figure:
    """Horizontal bar chart comparing retain vs churn model confidence."""
    fig = go.Figure(
        go.Bar(
            x=[retain, churn],
            y=["Likely to stay", "Churn risk"],
            orientation="h",
            marker_color=["#22c55e", "#ef4444"],
            text=[f"{retain:.1%}", f"{churn:.1%}"],
            textposition="auto",
        )
    )
    fig.update_layout(
        title={"text": "Model output distribution", "font": {"size": 15, "color": "#e8edf4"}},
        xaxis_title="Probability",
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(autorange="reversed"),
        height=220,
        margin=dict(l=10, r=10, t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e8edf4", "family": "DM Sans, sans-serif"},
    )
    return fig


def _render_prediction_hero(result: PredictionResult, hero_anim_class: str) -> None:
    """Large, bold outcome strip — dominant visual for executives and ops."""
    churn_word = "YES" if result.churn else "NO"
    churn_css = "result-hero-churn-yes" if result.churn else "result-hero-churn-no"
    sub = (
        "Elevated churn risk — prioritize retention workflows."
        if result.churn
        else "Within acceptable risk tolerance for standard lifecycle management."
    )
    prob_pct = f"{result.probability_churn * 100:.2f}".rstrip("0").rstrip(".")
    pill_class = f"pill-{result.risk_css_class}"
    st.markdown(
        f"""
        <div class="result-hero-wrap{hero_anim_class}">
            <div class="result-hero-grid">
                <div>
                    <div class="result-hero-label">Churn prediction</div>
                    <div class="result-hero-churn {churn_css}">{churn_word}</div>
                    <p class="result-hero-sub">{sub}</p>
                    <div class="result-hero-pills">
                        <span class="metric-pill {pill_class}">Risk tier · {result.risk_level}</span>
                        <span class="metric-pill" style="background:rgba(99,102,241,0.15);color:#c7d2fe;border-color:rgba(129,140,248,0.25);">
                            Class output · {result.churn_label}
                        </span>
                    </div>
                </div>
                <div class="result-hero-prob-block">
                    <div class="result-hero-label">Modeled churn probability</div>
                    <div class="result-hero-prob-val">{prob_pct}%</div>
                    <div class="result-hero-prob-cap">Calibrated score (0–100%)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recommended_action(result: PredictionResult, rec_anim_class: str) -> None:
    """Operational guidance aligned to telecom retention practice."""
    if result.risk_level == "Low":
        panel = "low"
        title = "Recommended action · lifecycle rhythm"
        bullets = (
            "Keep on <strong>standard</strong> nurture: usage tips, network quality notices, and optional add-on offers.",
            "No immediate <strong>retention desk</strong> escalation; monitor ARPU and support tickets quarterly.",
            "Use this segment for <strong>upsell</strong> tests (5G, home internet) where policy allows.",
        )
    elif result.risk_level == "Medium":
        panel = "med"
        title = "Recommended action · proactive save track"
        bullets = (
            "Route to <strong>retention specialist</strong> within 48 hours with save-ready offers (loyalty credit, plan review).",
            "Validate <strong>billing disputes</strong> and payment friction; offer autopay or paperless incentives.",
            "Schedule <strong>human outreach</strong> (voice/SMS) before contract window or competitive promo exposure.",
        )
    else:
        panel = "high"
        title = "Recommended action · critical intervention"
        bullets = (
            "<strong>Same-day</strong> supervisor callback and documented save path (contract adjustment, temporary rate relief).",
            "Flag account for <strong>revenue assurance</strong> (usage vs charges) and fraud pattern check if applicable.",
            "Prepare <strong>win-back</strong> contingency if disconnect initiated; log outcome in CRM for model feedback.",
        )
    items = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(
        f"""
        <div class="recommended-panel {panel}{rec_anim_class}">
            <h4>📋 {title}</h4>
            <ul style="margin:0;padding-left:1.15rem;color:rgba(203,213,225,0.96);font-size:0.88rem;line-height:1.55;">
                {items}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_results_card(result: PredictionResult) -> None:
    """Analytics dashboard panel: hero outcome, guidance, gauge, and distribution."""
    play_anim = st.session_state.pop("_play_prediction_animation", False)
    hero_anim = " result-entry-anim" if play_anim else ""
    rec_anim = " rec-action-anim" if play_anim else ""

    with st.container(border=True):
        st.markdown(
            '<p class="dash-charts-title" style="margin-top:0.15rem;">Scoring output</p>',
            unsafe_allow_html=True,
        )
        _render_prediction_hero(result, hero_anim_class=hero_anim)
        _render_recommended_action(result, rec_anim_class=rec_anim)

        st.markdown(
            '<p class="dash-charts-title">Exposure & distribution</p>',
            unsafe_allow_html=True,
        )
        g1, g2 = st.columns((1.12, 1))
        with g1:
            st.plotly_chart(
                _gauge_figure(result.probability_churn),
                use_container_width=True,
            )
        with g2:
            st.markdown("**Live probability track**")
            st.caption("Relative to enterprise risk bands used in reporting.")
            st.progress(
                min(max(result.probability_churn, 0.0), 1.0),
                text=f"{result.probability_churn:.1%} modeled churn likelihood",
            )
            st.plotly_chart(
                _confidence_bar_figure(result.probability_retain, result.probability_churn),
                use_container_width=True,
            )


def _render_empty_state() -> None:
    with st.container(border=True):
        st.markdown("#### Scorecard ready")
        st.info(
            f"Build a **360° subscriber view** in the **{BRAND_PRODUCT_LINE}** sidebar, "
            "then select **Run scoring** to generate the executive scorecard, "
            "recommended actions, and distribution charts in this workspace."
        )


def _render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer-bar">
            <div class="footer-bar-inner">
                <div class="footer-brand">{BRAND_COMPANY}</div>
                <div class="footer-meta">
                    {BRAND_PRODUCT_LINE} · {APP_TITLE} · v{PROJECT_VERSION}
                </div>
                <div class="footer-legal">{FOOTER_LEGAL}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# App entry
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} · {BRAND_COMPANY}",
        page_icon="📉",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    _init_session_state()

    _render_header()

    try:
        model = load_model()
        scaler = load_scaler()
    except FileNotFoundError as e:
        st.error(f"**Configuration error:** {e}")
        st.stop()
    except Exception as e:
        st.error("**Could not load model artifacts.** Please verify `model.pkl` and `scaler.pkl`.")
        with st.expander("Technical details"):
            st.exception(e)
        st.stop()

    predict_clicked, reset_clicked = _render_sidebar_inputs()

    if reset_clicked:
        _reset_inputs()
        st.rerun()

    if predict_clicked:
        raw = _collect_raw_fields_from_state()
        errors = validate_inputs(
            phone_service=raw["phone_service"],
            multiple_lines=raw["multiple_lines"],
            internet_service=raw["internet_service"],
            online_security=raw["online_security"],
            online_backup=raw["online_backup"],
            device_protection=raw["device_protection"],
            tech_support=raw["tech_support"],
            streaming_tv=raw["streaming_tv"],
            streaming_movies=raw["streaming_movies"],
            monthly_charges=raw["monthly_charges"],
            total_charges=raw["total_charges"],
        )
        if errors:
            for msg in errors:
                st.error(msg)
        else:
            with st.spinner("Calibrating churn model — engineering features, scaling, and scoring…"):
                try:
                    X_scaled, feature_row = preprocess_input(raw, scaler)
                    result = predict(model, X_scaled)
                    st.session_state["prediction"] = result
                    st.session_state["last_feature_row"] = feature_row
                    st.session_state["has_prediction"] = True
                    st.session_state["_play_prediction_animation"] = True
                except ValueError as e:
                    st.error(f"**Invalid input for preprocessing:** {e}")
                except Exception as e:
                    st.error("**Prediction failed.** Check inputs and artifact compatibility.")
                    with st.expander("Technical details"):
                        st.exception(e)

    if st.session_state.get("has_prediction") and "prediction" in st.session_state:
        _render_results_card(st.session_state["prediction"])
        with st.expander("🔬 Engineered features (raw, pre-scale)", expanded=False):
            fr = st.session_state.get("last_feature_row")
            if fr:
                derived = {
                    "ChargeRatio": fr["ChargeRatio"],
                    "AvgMonthlySpend": fr["AvgMonthlySpend"],
                    "TenureBand": fr["TenureBand"],
                    "ServiceCount": fr["ServiceCount"],
                    "HighRiskFlag": fr["HighRiskFlag"],
                    "UltraLoyal": fr["UltraLoyal"],
                }
                st.json(derived)
                st.json({k: float(v) for k, v in fr.items()})
    else:
        _render_empty_state()

    _render_footer()


if __name__ == "__main__":
    main()
