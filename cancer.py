import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# PAGE CONFIG
st.set_page_config(
    page_title="Cancer Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# FULL CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #060b14 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] > .main {
    background: #060b14 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1200px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Hero ── */
.hero-wrap {
    position: relative;
    padding: 3.5rem 3rem 2.5rem;
    margin-bottom: 2rem;
    border-radius: 20px;
    overflow: hidden;
    background: linear-gradient(135deg, #0d1b2e 0%, #0a1628 50%, #0d1b2e 100%);
    border: 1px solid rgba(56, 189, 248, 0.12);
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -40px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38bdf8;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    color: #f0f6ff;
    margin-bottom: 0.75rem;
}
.hero-title span { color: #38bdf8; }
.hero-sub {
    color: #7090b0;
    font-size: 0.95rem;
    max-width: 540px;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.hero-stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
}
.hero-stat-lbl {
    font-size: 0.72rem;
    color: #4a6a8a;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-top: 1px;
}

/* ── Section label ── */
.section-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.4rem;
    margin-top: 1.8rem;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #cde;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(56,189,248,0.1);
}

/* ── Input Cards ── */
.input-card {
    background: #0d1829;
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 14px;
    padding: 1.5rem 1.5rem 0.5rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.3s;
}
.input-card:hover { border-color: rgba(56,189,248,0.22); }

/* ── Streamlit widget overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
.stSelectbox > div > div,
[data-baseweb="select"] > div {
    background: #0a1525 !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 10px !important;
    color: #cde !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stNumberInput"] input:focus,
.stSelectbox > div > div:focus-within {
    border-color: rgba(56,189,248,0.45) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
}

[data-testid="stSlider"] > div > div > div {
    background: #38bdf8 !important;
}

label, .stRadio label, .stSelectbox label,
[data-testid="stSlider"] label {
    color: #7090b0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}

/* ── Analyze Button ── */
.stButton > button {
    width: 100% !important;
    padding: 1rem 2rem !important;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 24px rgba(14,165,233,0.25) !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    box-shadow: 0 0 40px rgba(14,165,233,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Divider ── */
hr { border-color: rgba(56,189,248,0.08) !important; margin: 1.5rem 0 !important; }

/* ── Result Cards ── */
.result-positive {
    background: linear-gradient(135deg, #022c22 0%, #064e3b 100%);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: glowGreen 3s ease-in-out infinite;
}
.result-positive::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #10b981, #34d399, #10b981);
    animation: shimmer 2s linear infinite;
    background-size: 200% 100%;
}
.result-negative {
    background: linear-gradient(135deg, #2d0a0a 0%, #450a0a 100%);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 18px;
    padding: 2rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: glowRed 3s ease-in-out infinite;
}
.result-negative::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #ef4444, #f87171, #ef4444);
    animation: shimmer 2s linear infinite;
    background-size: 200% 100%;
}
@keyframes glowGreen {
    0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.15); }
    50% { box-shadow: 0 0 40px rgba(16,185,129,0.3); }
}
@keyframes glowRed {
    0%, 100% { box-shadow: 0 0 20px rgba(239,68,68,0.15); }
    50% { box-shadow: 0 0 40px rgba(239,68,68,0.3); }
}
@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-subtitle { font-size: 0.85rem; opacity: 0.75; margin-bottom: 1rem; }
.result-desc { font-size: 0.9rem; opacity: 0.85; line-height: 1.6; max-width: 480px; margin: 0 auto; }

/* ── Confidence Meter ── */
.conf-wrap { margin: 1.5rem 0; }
.conf-label {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.5rem;
}
.conf-label-text { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #7090b0; }
.conf-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 700; }
.conf-track {
    height: 10px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s ease;
    position: relative;
}
.conf-fill::after {
    content: '';
    position: absolute; top: 0; right: 0; bottom: 0; width: 30px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3));
    animation: sweep 2s ease-in-out infinite;
}
@keyframes sweep { 0%, 100% { opacity: 0; } 50% { opacity: 1; } }

/* ── Metric Cards ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.9rem;
    margin-top: 1rem;
}
.metric-card {
    background: #0d1829;
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    transition: border-color 0.3s, transform 0.2s;
}
.metric-card:hover { border-color: rgba(56,189,248,0.25); transform: translateY(-2px); }
.metric-card-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
.metric-card-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.15rem;
}
.metric-card-lbl { font-size: 0.72rem; color: #4a6a8a; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Severity Badge ── */
.severity-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1rem; align-items: center; }
.sev-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.sev-low { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.sev-mod { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.sev-high { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.sev-crit { background: rgba(220,38,38,0.2); color: #fca5a5; border: 1px solid rgba(220,38,38,0.35); }

/* ── Recommendations ── */
.rec-list { list-style: none; padding: 0; margin: 0.8rem 0 0; }
.rec-list li {
    padding: 0.65rem 0.9rem;
    background: rgba(56,189,248,0.04);
    border-left: 3px solid rgba(56,189,248,0.3);
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #a0bad0;
    margin-bottom: 0.5rem;
    line-height: 1.5;
}
.rec-list li span { color: #38bdf8; font-weight: 600; }

/* ── Summary Table ── */
.summary-table { width: 100%; border-collapse: collapse; margin-top: 0.8rem; }
.summary-table td { padding: 0.55rem 0.75rem; font-size: 0.83rem; border-bottom: 1px solid rgba(56,189,248,0.06); }
.summary-table td:first-child { color: #4a6a8a; font-weight: 500; width: 42%; }
.summary-table td:last-child { color: #c8dff0; font-weight: 500; }

/* ── Survival Bar ── */
.survival-bar-wrap { margin: 0.8rem 0; }
.survival-label { font-size: 0.8rem; color: #7090b0; margin-bottom: 0.4rem; display: flex; justify-content: space-between; }
.survival-track { height: 8px; background: rgba(255,255,255,0.05); border-radius: 100px; overflow: hidden; }
.survival-fill { height: 100%; border-radius: 100px; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #2a4a6a;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
}
.footer a { color: #38bdf8; text-decoration: none; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #38bdf8 !important; }

/* ── Radio ── */
[data-testid="stRadio"] label { color: #7090b0 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🧬Clinical Decision Support</div>
  <div class="hero-title">Next-Generation Cancer Risk &<span> Prognosis Intelligence</span></div>
  <div class="hero-sub">Advanced machine-learning platform for cancer patient outcome analysis. Input clinical parameters to receive AI-driven survival outlook, risk stratification, and evidence-based care recommendations.</div>
  <div class="hero-stats">
    <div><div class="hero-stat-val">9</div><div class="hero-stat-lbl">Cancer Types</div></div>
    <div><div class="hero-stat-val">7</div><div class="hero-stat-lbl">Treatment Modalities</div></div>
    <div><div class="hero-stat-val">9</div><div class="hero-stat-lbl">States Covered</div></div>
    <div><div class="hero-stat-val">ML</div><div class="hero-stat-lbl">Model Engine</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model   = joblib.load('logistic.pkl')
    scaler  = joblib.load('Logistic Scaler.pkl')
    columns = joblib.load('Logistic Columns.pkl')
    return model, scaler, columns

try:
    model, scaler, columns = load_assets()
except Exception:
    st.error("⚠️ Model files not found. Please ensure `logistic.pkl`, `Logistic Scaler.pkl`, and `Logistic Columns.pkl` are in the project directory.")
    st.stop()

# ──────────────────────────────────────────────
# INPUT FORM  (two-column layout)
# ──────────────────────────────────────────────
st.markdown('<div class="section-eyebrow">Patient Data Input</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Clinical Parameters</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("**🧑 Demographics**")
    age    = st.number_input("Age (years)", 1, 100, 45)
    gender = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
    state  = st.selectbox("State", ["Delhi", "Maharashtra", "Gujarat", "Karnataka",
                                     "West Bengal", "Tamil Nadu", "Telangana",
                                     "Kerala", "Chandigarh"])
    city   = st.selectbox("City", ["New Delhi", "Mumbai", "Kolkata", "Chennai",
                                    "Hyderabad", "Bengaluru", "Ahmedabad",
                                    "Thiruvananthapuram", "Chandigarh"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("**🏥 Diagnosis Details**")
    cancer_type = st.selectbox("Cancer Type", ["Breast Cancer", "Lung Cancer", "Leukemia",
                                                "Oral Cancer", "Stomach Cancer", "Cervical Cancer",
                                                "Colorectal Cancer", "Prostate Cancer", "Ovarian Cancer"])
    stage      = st.selectbox("Clinical Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
    survival   = st.slider("Survival Months Recorded", 0.0, 60.0, 30.0, step=0.5)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("**📅 Diagnosis Date**")
    diag_year  = st.number_input("Year",  2022, 2025, 2023)
    diag_month = st.number_input("Month", 1,    12,   6)
    diag_day   = st.number_input("Day",   1,    31,   15)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("**⚕️ Clinical Setting**")
    hospital  = st.selectbox("Hospital / Institute", [
        "Rajiv Gandhi Cancer Institute", "Tata Memorial Hospital",
        "Tata Medical Center", "AIIMS",
        "Kidwai Memorial Institute of Oncology", "Regional Cancer Centre",
        "Apollo Hospital", "Gujarat Cancer Research Institute", "PGIMER"
    ])
    treatment = st.selectbox("Treatment Protocol", [
        "Surgery", "Chemotherapy", "Radiation", "Palliative Care",
        "Targeted Therapy", "Surgery + Chemotherapy", "Chemo + Radiation"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ANALYZE BUTTON
# ──────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run = st.button("🧬  RUN ONCOLOGY ANALYSIS")

# ──────────────────────────────────────────────
# HELPER: RULE-BASED CONFIDENCE SCORE
# ──────────────────────────────────────────────
def compute_confidence(stage_str, survival_months, age, treatment_str):
    """
    Custom confidence score (0–100) derived from clinical factors,
    NOT from raw model probability alone.
    Weights: Stage (40pts) + Survival Months (30pts) + Age (15pts) + Treatment (15pts)
    """
    stage_map = {"Stage I": 40, "Stage II": 30, "Stage III": 16, "Stage IV": 4}
    stage_pts = stage_map.get(stage_str, 20)

    # Survival months: 0–60 scaled to 0–30
    surv_pts = min(survival_months / 60.0, 1.0) * 30

    # Age penalty: younger = better prognosis weight
    if age < 40:   age_pts = 15
    elif age < 55: age_pts = 12
    elif age < 70: age_pts = 8
    else:          age_pts = 4

    # Treatment modifier
    aggressive = ["Surgery + Chemotherapy", "Chemo + Radiation", "Targeted Therapy", "Surgery"]
    trt_pts = 15 if treatment_str in aggressive else 8

    raw = stage_pts + surv_pts + age_pts + trt_pts   # 0–100
    return round(min(raw, 99), 1)

def stage_severity(stage_str):
    return {
        "Stage I":   ("LOW", "sev-low",  "Early-stage. High curative potential."),
        "Stage II":  ("MODERATE", "sev-mod",  "Locally advanced. Good treatment response expected."),
        "Stage III": ("HIGH", "sev-high", "Regionally spread. Aggressive multimodal therapy recommended."),
        "Stage IV":  ("CRITICAL", "sev-crit", "Metastatic disease. Palliative and targeted options prioritised."),
    }.get(stage_str, ("UNKNOWN", "sev-mod", ""))

def survival_band(months):
    if months >= 48: return "Excellent (≥ 48 mo)", "#10b981"
    if months >= 30: return "Moderate (30–47 mo)", "#fbbf24"
    if months >= 12: return "Limited (12–29 mo)", "#f97316"
    return "Short-term (< 12 mo)", "#ef4444"

def recommendations(pred, stage_str, cancer_type, treatment_str):
    if pred == 1:  # Positive
        recs = [
            "<span>Continue</span> current treatment protocol and schedule 3-month imaging review.",
            f"<span>{cancer_type}</span>: coordinate multidisciplinary tumour board review.",
            "<span>Monitor</span> CBC, liver function, and tumour markers every 6 weeks.",
            "<span>Nutritional support</span> and physiotherapy referral recommended for treatment tolerance.",
            "Discuss <span>survivorship plan</span> including psychological support and lifestyle modifications.",
        ]
    else:  # High Risk
        recs = [
            f"<span>Urgent MDT review</span> — escalate care plan for {stage_str} presentation.",
            f"<span>Treatment intensification</span>: evaluate suitability for {treatment_str} escalation or clinical trial.",
            "<span>Palliative care</span> integration and goals-of-care discussion with patient and family.",
            "<span>Pain and symptom management</span> pathway to be initiated without delay.",
            "Consider <span>genomic profiling</span> for targeted therapy eligibility.",
        ]
    return recs

# ──────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────
if run:
    with st.spinner("Running oncology analysis model…"):
        time.sleep(1.2)

        # ── Preprocessing ──
        input_df = pd.DataFrame(0, index=[0], columns=columns)
        input_df['Age']             = age
        input_df['Survival_Months'] = survival
        input_df['Diagnosis_Year']  = diag_year
        input_df['Diagnosis_Month'] = diag_month
        input_df['Diagnosis_Day']   = diag_day
        input_df['Stage']           = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}[stage]

        for col in [f'Gender_{gender}', f'State_{state}', f'City_{city}',
                    f'Hospital_Name_{hospital}', f'Cancer_Type_{cancer_type}',
                    f'Treatment_Type_{treatment}']:
            if col in columns:
                input_df[col] = 1

        num_cols = ['Age', 'Survival_Months', 'Diagnosis_Year', 'Diagnosis_Month', 'Diagnosis_Day']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        pred = model.predict(input_df)[0]
        conf = compute_confidence(stage, survival, age, treatment)

    # ── Divider ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">Analysis Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Output</div>', unsafe_allow_html=True)

    # ── Result Card ──
    if pred == 1:
        conf_color = "#10b981"
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-icon">🟢</div>
            <div class="result-title" style="color:#34d399;">Survival Outlook: Positive</div>
            <div class="result-subtitle">Model predicts favourable patient outcome</div>
            <div class="conf-wrap">
                <div class="conf-label">
                    <span class="conf-label-text">Clinical Confidence Score</span>
                    <span class="conf-value" style="color:{conf_color};">{conf}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{conf}%; background: linear-gradient(90deg, #059669, #34d399);"></div>
                </div>
            </div>
            <div class="result-desc">
                Based on the patient's stage, recorded survival months, age profile, and treatment protocol, the model indicates a <strong>positive survival trajectory</strong>. Continued adherence to the treatment plan is strongly advised.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        conf_color = "#ef4444"
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-icon">🔴</div>
            <div class="result-title" style="color:#f87171;">Survival Outlook: High Risk</div>
            <div class="result-subtitle">Model identifies elevated mortality risk</div>
            <div class="conf-wrap">
                <div class="conf-label">
                    <span class="conf-label-text">Clinical Confidence Score</span>
                    <span class="conf-value" style="color:{conf_color};">{conf}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{conf}%; background: linear-gradient(90deg, #dc2626, #f87171);"></div>
                </div>
            </div>
            <div class="result-desc">
                Clinical indicators suggest a <strong>high-risk outcome profile</strong>. Immediate escalation to a multidisciplinary tumour board and review of the current care plan is strongly recommended.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics Row ──
    sev_label, sev_cls, sev_desc = stage_severity(stage)
    surv_label, surv_color       = survival_band(survival)
    age_group = "Paediatric / Young" if age < 30 else ("Middle-aged" if age < 60 else "Elderly (≥60)")
    treat_type = "Curative Intent" if treatment in ["Surgery", "Surgery + Chemotherapy", "Chemo + Radiation", "Targeted Therapy"] else "Palliative / Supportive"

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-card-icon">📍</div>
            <div class="metric-card-val">{stage}</div>
            <div class="metric-card-lbl">Cancer Stage</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-icon">⏱️</div>
            <div class="metric-card-val">{int(survival)} mo</div>
            <div class="metric-card-lbl">Survival Recorded</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-icon">🎂</div>
            <div class="metric-card-val">{age} yrs</div>
            <div class="metric-card-lbl">Patient Age</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-icon">💊</div>
            <div class="metric-card-val">{treat_type}</div>
            <div class="metric-card-lbl">Treatment Type</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-icon">📊</div>
            <div class="metric-card-val">{conf}%</div>
            <div class="metric-card-lbl">Confidence Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-icon">🏥</div>
            <div class="metric-card-val">{'Tertiary' if 'AIIMS' in hospital or 'Tata' in hospital else 'Specialty'}</div>
            <div class="metric-card-lbl">Hospital Level</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Advanced Insights (two columns) ──
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="section-eyebrow">Stage Severity</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0d1829; border:1px solid rgba(56,189,248,0.1); border-radius:12px; padding:1.2rem 1.4rem;">
            <div class="severity-row">
                <span class="sev-badge {sev_cls}">{sev_label}</span>
                <span style="font-size:0.82rem; color:#7090b0;">{sev_desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-eyebrow" style="margin-top:1.4rem;">Survival Analysis</div>', unsafe_allow_html=True)
        surv_pct = min(int(survival / 60 * 100), 100)
        st.markdown(f"""
        <div style="background:#0d1829; border:1px solid rgba(56,189,248,0.1); border-radius:12px; padding:1.2rem 1.4rem;">
            <div class="survival-bar-wrap">
                <div class="survival-label">
                    <span>Survival Duration Band</span>
                    <span style="color:{surv_color}; font-weight:600;">{surv_label}</span>
                </div>
                <div class="survival-track">
                    <div class="survival-fill" style="width:{surv_pct}%; background:{surv_color};"></div>
                </div>
            </div>
            <div style="font-size:0.8rem; color:#4a6a8a; margin-top:0.5rem;">
                {int(survival)} months recorded out of 60-month follow-up window ({surv_pct}% of tracking period)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-eyebrow">Patient Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0d1829; border:1px solid rgba(56,189,248,0.1); border-radius:12px; padding:1.2rem 1.4rem;">
            <table class="summary-table">
                <tr><td>Cancer Type</td><td>{cancer_type}</td></tr>
                <tr><td>Patient Profile</td><td>{age_group} · {gender}</td></tr>
                <tr><td>Diagnosis Date</td><td>{diag_day:02d}/{diag_month:02d}/{diag_year}</td></tr>
                <tr><td>Treating Centre</td><td>{hospital}</td></tr>
                <tr><td>Protocol</td><td>{treatment}</td></tr>
                <tr><td>Location</td><td>{city}, {state}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown('<div class="section-eyebrow" style="margin-top:1.8rem;">Clinical Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evidence-Based Care Guidance</div>', unsafe_allow_html=True)
    rec_items = recommendations(pred, stage, cancer_type, treatment)
    items_html = "".join(f"<li>{r}</li>" for r in rec_items)
    st.markdown(f'<ul class="rec-list">{items_html}</ul>', unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("""
    <div style="margin-top:2rem; padding:1rem 1.2rem; background:rgba(245,158,11,0.05); border:1px solid rgba(245,158,11,0.15); border-radius:10px; font-size:0.78rem; color:#7090b0; line-height:1.6;">
        ⚠️ <strong style="color:#fbbf24;">Clinical Disclaimer:</strong> All predictions must be reviewed and validated by a qualified oncologist before informing any clinical decision. Model outputs do not constitute a medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="footer">
    OncoPulse AI · Clinical Decision Support Platform &nbsp;|&nbsp;
    Built with Logistic Regression + Streamlit &nbsp;|&nbsp;
    <span style="color:#2a4a6a;">For Research & Portfolio Use Only</span>
</div>
""", unsafe_allow_html=True)