"""Student Attendance Risk Predictor - Streamlit app (educational prototype).

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from src.data_utils import (
    DATA_DICTIONARY,
    DEFAULT_SAMPLE_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    attendance_pct,
    generate_synthetic_data,
    load_csv,
    template_csv,
    validate_dataframe,
    validate_single,
)
from src.evaluation import confusion_df, plot_confusion_matrix
from src.labeling import AT_RISK, DEFAULT_THRESHOLD, add_risk_labels, classify_attendance
from src.modeling import (
    BASELINE_MAJORITY,
    BASELINE_RULE,
    MODEL_NAMES,
    ModelingError,
    feature_importance,
    predict_dataframe,
    run_experiment,
)

st.set_page_config(
    page_title="Attendance Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp { background-color: #F8FAFC; }
.app-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
    padding: 2rem 2.5rem 1.75rem; border-radius: 12px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(37,99,235,0.20);
}
.app-header h1 { color:#FFFFFF; font-size:1.95rem; font-weight:700; margin:0 0 0.35rem; letter-spacing:-0.02em; }
.app-header p  { color:#BFDBFE; font-size:0.92rem; margin:0; }
.app-header .badge {
    display:inline-block; background:rgba(255,255,255,0.15); color:#E0F2FE;
    font-size:0.75rem; padding:2px 10px; border-radius:20px; margin-top:0.6rem;
    border:1px solid rgba(255,255,255,0.25); font-weight:500; letter-spacing:0.03em;
}
.disclaimer-strip {
    background:#FFFBEB; border:1px solid #FDE68A; border-left:4px solid #F59E0B;
    border-radius:8px; padding:0.75rem 1rem; margin-bottom:1.25rem;
    font-size:0.84rem; color:#78350F; line-height:1.55;
}
[data-testid="metric-container"] {
    background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
    padding:1rem 1.25rem !important; box-shadow:0 1px 4px rgba(0,0,0,0.06);
    transition:box-shadow 0.2s ease;
}
[data-testid="metric-container"]:hover { box-shadow:0 4px 14px rgba(0,0,0,0.10); }
[data-testid="metric-container"] > label {
    font-size:0.72rem !important; font-weight:600 !important;
    text-transform:uppercase; letter-spacing:0.06em; color:#64748B !important;
}
[data-testid="metric-container"] > div:nth-child(2) {
    font-size:1.6rem !important; font-weight:700 !important; color:#0F172A !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap:0px; background:#F1F5F9; border-radius:10px; padding:4px; border-bottom:none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius:7px; padding:0.45rem 1.1rem; font-size:0.85rem; font-weight:500;
    color:#64748B; border:none; background:transparent; transition:all 0.15s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color:#1E40AF; background:rgba(255,255,255,0.7); }
[data-testid="stTabs"] [aria-selected="true"] {
    background:#FFFFFF !important; color:#1E40AF !important; font-weight:600 !important;
    box-shadow:0 1px 6px rgba(0,0,0,0.10);
}
.section-card {
    background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
    padding:1.4rem 1.6rem; margin-bottom:1rem; box-shadow:0 1px 4px rgba(0,0,0,0.05);
}
.section-title {
    font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em;
    color:#94A3B8; margin-bottom:0.85rem; padding-bottom:0.55rem; border-bottom:1px solid #F1F5F9;
}
.pred-card-risk {
    background:linear-gradient(135deg,#FEF2F2 0%,#FFF5F5 100%); border:1.5px solid #FECACA;
    border-left:5px solid #EF4444; border-radius:10px; padding:1.25rem 1.5rem; margin:0.75rem 0;
}
.pred-card-safe {
    background:linear-gradient(135deg,#F0FDF4 0%,#F7FFF8 100%); border:1.5px solid #BBF7D0;
    border-left:5px solid #22C55E; border-radius:10px; padding:1.25rem 1.5rem; margin:0.75rem 0;
}
.pred-card-risk h3 { color:#B91C1C; font-size:1.15rem; font-weight:700; margin:0 0 0.3rem; }
.pred-card-safe h3 { color:#15803D; font-size:1.15rem; font-weight:700; margin:0 0 0.3rem; }
.pred-card-risk p, .pred-card-safe p { font-size:0.875rem; margin:0; color:#374151; }
.prob-bar-wrap { margin:0.85rem 0 0.3rem; }
.prob-bar-label { font-size:0.78rem; font-weight:600; color:#475569; margin-bottom:0.35rem; }
.prob-bar-outer { background:#E5E7EB; border-radius:6px; height:10px; overflow:hidden; }
.prob-bar-inner-risk { background:linear-gradient(90deg,#F97316,#EF4444); height:100%; border-radius:6px; }
.prob-bar-inner-safe { background:linear-gradient(90deg,#34D399,#22C55E); height:100%; border-radius:6px; }
.calc-box {
    background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px;
    padding:0.85rem 1.1rem; font-size:0.86rem; color:#475569; line-height:1.7;
    margin:0.6rem 0; font-family:'Courier New',monospace;
}
.calc-highlight { color:#1D4ED8; font-weight:700; }
[data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #E2E8F0; }
[data-testid="stDataFrame"] { border:1px solid #E2E8F0 !important; border-radius:8px !important; overflow:hidden; }
[data-testid="stAlert"] { border-radius:8px; font-size:0.875rem; }
[data-testid="stExpander"] { border:1px solid #E2E8F0 !important; border-radius:8px !important; background:#FFFFFF; }
</style>
""", unsafe_allow_html=True)

DISCLAIMER = (
    "⚠️ <strong>Educational prototype.</strong> All data are synthetic. Risk labels are a simple threshold rule. "
    "Do not use these outputs for consequential decisions about real students without proper validation, "
    "institutional policy checks, and human review."
)

# --------------------------------------------------------------------------
# Cached data / model loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, str]:
    """Bundled sample CSV; falls back to in-memory synthetic data if it is missing."""
    try:
        return load_csv(DEFAULT_SAMPLE_PATH), ""
    except (FileNotFoundError, ValueError) as exc:
        return generate_synthetic_data(), f"{exc} Using freshly generated synthetic data instead."


@st.cache_resource(show_spinner="Training models — please wait...")
def get_experiment(threshold: float):
    df, _ = load_data()
    return run_experiment(df, threshold)


# --------------------------------------------------------------------------
# App Header
# --------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🎓 Student Attendance Risk Predictor</h1>
    <p>Pune College &nbsp;·&nbsp; ML Unit 3 &nbsp;·&nbsp; Attendance Risk Classification</p>
    <span class="badge">EDUCATIONAL PROTOTYPE &nbsp;·&nbsp; SYNTHETIC DATA ONLY</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f'<div class="disclaimer-strip">{DISCLAIMER}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    threshold = st.number_input(
        "Attendance Threshold (%)",
        min_value=1.0, max_value=100.0,
        value=float(DEFAULT_THRESHOLD), step=1.0,
        help=(
            "Students BELOW this value are classified as 'At Risk'. "
            "75% is an example only — verify your institution's actual rule."
        ),
    )

    st.markdown("""
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
         padding:0.75rem 0.9rem;margin:0.6rem 0;font-size:0.82rem;color:#1E40AF;line-height:1.6;">
    <strong>ℹ️ Approach A (Current Status)</strong><br>
    <code>At Risk</code> = attendance % &lt; threshold<br>
    This is <em>rule-based classification</em>, not future-risk prediction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.78rem;color:#64748B;margin-top:0.75rem;">
    <strong>Positive class:</strong> At Risk<br>
    <strong>Boundary:</strong> Exactly at threshold = <em>Not at Risk</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#94A3B8;line-height:1.6;">
    📊 600 synthetic students (seed 42)<br>
    🤖 Logistic Reg · Decision Tree · Random Forest<br>
    🔀 75/25 stratified split
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Data + experiment
# --------------------------------------------------------------------------
raw_df, load_note = load_data()
if load_note:
    st.error(load_note)

with st.spinner("Training models on sample data…"):
    try:
        experiment = get_experiment(float(threshold))
    except ModelingError as exc:
        st.error(f"**Could not train models:** {exc}")
        st.info("Try adjusting the attendance threshold in the sidebar so both risk classes are present.")
        st.stop()

validation = validate_dataframe(raw_df)
labeled = add_risk_labels(validation.valid_df, threshold)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_dash, tab_pred, tab_batch, tab_eval, tab_meth = st.tabs([
    "📊  Dashboard",
    "🧮  Single Prediction",
    "📁  Batch Prediction",
    "🧪  Model Evaluation",
    "📘  Dataset & Methodology",
])

# ══════════════════════════════════════════════════════════════════════════
# 1. Dashboard
# ══════════════════════════════════════════════════════════════════════════
with tab_dash:
    n = len(labeled)
    n_risk = int((labeled[LABEL_COLUMN] == AT_RISK).sum())
    n_safe = n - n_risk
    mean_att = labeled["attendance_pct"].mean()
    counts = labeled[LABEL_COLUMN].value_counts().reindex(["Not at Risk", AT_RISK], fill_value=0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", f"{n:,}")
    c2.metric("Mean attendance", f"{mean_att:.1f}%")
    c3.metric("At Risk", f"{n_risk:,}", f"{n_risk / n:.1%} of students", delta_color="inverse")
    c4.metric("Not at Risk", f"{n_safe:,}", f"{n_safe / n:.1%} of students", delta_color="off")
    c5.metric("Threshold", f"{threshold:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-strip" style="background:#EFF6FF;border-color:#BFDBFE;
         border-left-color:#3B82F6;color:#1E3A5F;">
    📂 <strong>Synthetic dataset:</strong> all 600 records are fictional and generated by code.
    They are <em>not</em> real college records and contain no personal information.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown('<div class="section-title">Attendance Distribution (5-point bins)</div>', unsafe_allow_html=True)
        bins = list(range(0, 105, 5))
        bin_labels = [f"{b}-{b + 5}%" for b in bins[:-1]]
        binned = pd.cut(labeled["attendance_pct"], bins=bins, labels=bin_labels, right=False, include_lowest=True)
        chart_data = binned.value_counts().reindex(bin_labels, fill_value=0).rename("Students")
        st.bar_chart(chart_data, height=290, color="#3B82F6")
        st.caption(f"Threshold ({threshold:.0f}%) is the boundary for At Risk classification.")

    with right:
        st.markdown('<div class="section-title">Risk Class Distribution</div>', unsafe_allow_html=True)
        risk_chart = counts.rename("Students")
        st.bar_chart(risk_chart, height=290)
        st.caption(f"At Risk: {n_risk}/{n} students ({n_risk/n:.1%}) at {threshold:.0f}% threshold.")

    with st.expander("🔍 Dataset preview and class balance details"):
        col_prev, col_bal = st.columns([2, 1])
        with col_prev:
            st.markdown("**First 50 rows (labeled at current threshold)**")
            st.dataframe(labeled.head(50), hide_index=True, width='stretch')
        with col_bal:
            st.markdown("**Class balance**")
            bal = (counts / counts.sum()).round(4)
            for cls, pct_val in bal.items():
                color = "#EF4444" if cls == AT_RISK else "#22C55E"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:0.45rem 0.6rem;border-radius:6px;border:1px solid #E2E8F0;'
                    f'margin-bottom:0.4rem;background:#FAFAFA;">'
                    f'<span style="font-size:0.85rem;font-weight:500;">{cls}</span>'
                    f'<span style="color:{color};font-weight:700;">{pct_val:.1%}</span></div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════
# 2. Single Student Prediction
# ══════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("##### Predict Risk for One Student")
    st.caption(
        "Default values give a sample prediction (80 conducted, 52 attended → 65%). "
        "Adjust inputs to explore the model's behaviour."
    )

    model_name = st.selectbox(
        "Model", MODEL_NAMES,
        index=MODEL_NAMES.index("Random Forest"),
        key="single_model",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Required Inputs</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    conducted = c1.number_input(
        "Lectures Conducted", min_value=0, max_value=1000, value=80, step=1,
        help="Total lectures held in the current term (must be > 0).",
    )
    attended = c2.number_input(
        "Lectures Attended", min_value=0, max_value=1000, value=52, step=1,
        help="Number of lectures attended by this student.",
    )

    use_optional = st.checkbox(
        "Include optional features (previous attendance & recent absences)",
        value=False,
        help="If unchecked, the model uses the training-set median.",
    )
    prev_pct = recent = None
    if use_optional:
        st.markdown('<div class="section-title">Optional Inputs</div>', unsafe_allow_html=True)
        o1, o2 = st.columns(2, gap="medium")
        prev_pct = o1.number_input(
            "Previous Term Attendance (%)", min_value=0.0, max_value=100.0, value=72.0, step=0.5,
        )
        recent = o2.number_input(
            "Recent Absences (last 2 weeks)", min_value=0, max_value=100, value=3, step=1,
        )

    show_calc = st.checkbox("Show attendance calculation details", value=True)

    errors = validate_single(conducted, attended, prev_pct, recent)
    if errors:
        for msg in errors:
            st.error(f"**Input error:** {msg}")
    else:
        row = pd.DataFrame([{
            "lectures_conducted": conducted,
            "lectures_attended": attended,
            "previous_attendance_pct": np.nan if prev_pct is None else prev_pct,
            "recent_absences": np.nan if recent is None else recent,
        }])
        out = predict_dataframe(experiment.models[model_name], row, threshold).iloc[0]
        pct = float(out["attendance_pct"])
        pred, rule = out["predicted_label"], out["rule_based_label"]
        prob_risk = float(out["prob_at_risk"])

        is_at_risk = pred == AT_RISK
        card_class = "pred-card-risk" if is_at_risk else "pred-card-safe"
        icon = "🔴" if is_at_risk else "🟢"
        prob_bar_class = "prob-bar-inner-risk" if is_at_risk else "prob-bar-inner-safe"

        (st.error if is_at_risk else st.success)(
            f"### Predicted: {pred}  \n"
            f"Model estimate of P(At Risk) = {prob_risk:.1%} (Using {model_name} · Threshold: {threshold:.0f}%)"
        )

        st.markdown(f"""
        <div class="prob-bar-wrap" style="margin-top:-0.2rem;margin-bottom:0.75rem;">
            <div class="prob-bar-label">P(At Risk) = {prob_risk:.1%}</div>
            <div class="prob-bar-outer">
                <div class="{prob_bar_class}" style="width:{prob_risk*100:.1f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if show_calc:
            st.markdown(f"""
            <div class="calc-box">
            attendance_pct = {attended} / {conducted} &times; 100
                           = <span class="calc-highlight">{pct:.2f}%</span><br>
            threshold      = <span class="calc-highlight">{threshold:.0f}%</span> (example — verify with institution)<br>
            rule result    = <span class="calc-highlight">{rule}</span>
                             ({pct:.2f}% is {"below" if rule == AT_RISK else "at or above"} {threshold:.0f}%)
            </div>
            """, unsafe_allow_html=True)

        if pred != rule:
            st.warning(
                f"**Model-Rule Disagreement:** The {model_name} model predicts **{pred}** but the threshold rule "
                f"gives **{rule}**. Under Approach A the rule is the ground truth, so **{rule}** is the reference."
            )

        st.caption(
            "Why this output: the label is defined by the threshold rule; the model approximates "
            "that rule from lecture counts and optional features. It does not forecast future attendance."
        )

# ══════════════════════════════════════════════════════════════════════════
# 3. Batch Prediction
# ══════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("##### Batch Prediction from CSV")

    left_col, right_col = st.columns([2, 1], gap="large")
    with left_col:
        st.markdown("""
        <div class="section-card">
        <div class="section-title">How it works</div>
        Upload a CSV file with student lecture data. The app will validate each row, predict risk,
        and let you download the results.<br><br>
        <strong>Required columns:</strong> <code>lectures_conducted</code>, <code>lectures_attended</code><br>
        <strong>Optional columns:</strong> <code>student_id</code>, <code>previous_attendance_pct</code>, <code>recent_absences</code>
        </div>
        """, unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="section-title">CSV Template</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download CSV Template",
            data=template_csv(),
            file_name="attendance_template.csv",
            mime="text/csv",
            width='stretch',
        )
        st.caption("⚠️ Use fictional IDs only — never upload real personal data to public deployments.")

    st.markdown("---")

    col_model, _ = st.columns([1, 2])
    batch_model = col_model.selectbox(
        "Model for batch prediction", MODEL_NAMES,
        index=MODEL_NAMES.index("Random Forest"), key="batch_model",
    )

    upload = st.file_uploader("Upload CSV file", type=["csv"])

    if upload is not None:
        try:
            uploaded = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"**Could not read file as CSV:** {exc}")
            uploaded = None

        if uploaded is not None:
            with st.spinner("Validating and predicting…"):
                result = validate_dataframe(uploaded)

            if not result.ok and result.valid_df.empty and not len(result.invalid_df):
                for msg in result.schema_errors:
                    st.error(msg)
            else:
                if len(result.invalid_df):
                    st.warning(f"**{len(result.invalid_df)} row(s) rejected** and excluded from predictions.")
                    with st.expander(f"View {len(result.invalid_df)} rejected row(s) with reasons"):
                        st.dataframe(result.invalid_df, hide_index=True, width='stretch')

                if result.valid_df.empty:
                    for msg in result.schema_errors:
                        st.error(msg)
                else:
                    preds = predict_dataframe(experiment.models[batch_model], result.valid_df, threshold)
                    n_risk_pred = int((preds["predicted_label"] == AT_RISK).sum())
                    n_safe_pred = len(preds) - n_risk_pred

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rows Predicted", f"{len(preds):,}")
                    m2.metric("At Risk", f"{n_risk_pred:,}", delta_color="inverse")
                    m3.metric("Not at Risk", f"{n_safe_pred:,}", delta_color="off")

                    st.dataframe(preds, hide_index=True, width='stretch')
                    st.download_button(
                        label="⬇️ Download Predictions (CSV)",
                        data=preds.to_csv(index=False).encode("utf-8"),
                        file_name="attendance_risk_predictions.csv",
                        mime="text/csv",
                    )
    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 1rem;color:#94A3B8;
                    border:2px dashed #E2E8F0;border-radius:10px;background:#FAFAFA;margin-top:1rem;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">📂</div>
            <div style="font-size:1rem;font-weight:500;color:#64748B;">Upload a CSV file to get predictions</div>
            <div style="font-size:0.82rem;margin-top:0.35rem;">Download the template above to get started</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 4. Model Evaluation
# ══════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("##### Model Evaluation — Held-Out Test Set")

    st.markdown(f"""
    <div class="section-card" style="background:#F8FAFC;">
    <div class="section-title">Evaluation setup</div>
    <strong>Target:</strong> Approach A — <code>At Risk</code> if attendance % &lt; {threshold:.0f}%
    &nbsp;·&nbsp; <strong>Positive class:</strong> At Risk
    &nbsp;·&nbsp; <strong>Split:</strong> {len(experiment.X_train)} train / {len(experiment.X_test)} test rows
    (stratified 75/25, fixed random seed 42)
    </div>
    """, unsafe_allow_html=True)

    if experiment.n_rejected_rows:
        st.warning(f"**{experiment.n_rejected_rows} invalid row(s)** were excluded before training.")

    st.markdown('<div class="section-title" style="margin-top:1rem;">Performance Metrics</div>', unsafe_allow_html=True)
    m = experiment.metrics
    st.dataframe(
        m.style.format("{:.3f}").highlight_max(axis=0, color="#DCFCE7").highlight_min(
            axis=0, color="#FEF2F2", subset=["f1", "precision", "recall", "accuracy"]
        ),
        width='stretch',
    )

    best_model = m.loc[MODEL_NAMES, "f1"].idxmax()
    best_f1 = m.loc[best_model, "f1"]
    st.markdown(
        f'<div style="font-size:0.875rem;color:#166534;background:#DCFCE7;'
        f'border:1px solid #BBF7D0;border-radius:7px;padding:0.5rem 0.85rem;'
        f'display:inline-block;margin-bottom:0.75rem;">'
        f'🏆 Best model by At-Risk F1: <strong>{best_model}</strong> (F1 = {best_f1:.3f})</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "**Before quoting these scores:** The label is generated by a threshold rule on attendance counts. "
        "The rule baseline scores 100% **by construction**. High model scores show the model can reproduce "
        "that rule — not that it generalises, or predicts future attendance or academic outcomes."
    )

    st.markdown('<div class="section-title" style="margin-top:1.25rem;">Confusion Matrices (Rows = Actual, Cols = Predicted)</div>', unsafe_allow_html=True)
    cols = st.columns(len(MODEL_NAMES), gap="medium")
    for col, name in zip(cols, MODEL_NAMES):
        with col:
            st.pyplot(plot_confusion_matrix(experiment.confusion_matrices[name], name), width='stretch')

    with st.expander("📋 Baseline confusion matrices and per-class reports"):
        names = MODEL_NAMES + [BASELINE_RULE, BASELINE_MAJORITY]
        for name in names:
            st.markdown(f"**{name}**")
            c_left, c_right = st.columns(2, gap="medium")
            with c_left:
                st.dataframe(confusion_df(experiment.confusion_matrices[name]), width='stretch')
            with c_right:
                st.dataframe(
                    experiment.class_reports[name].style.format({
                        "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"
                    }),
                    width='stretch',
                )

    st.markdown('<div class="section-title" style="margin-top:1.25rem;">Feature Importance</div>', unsafe_allow_html=True)
    fi_col, _ = st.columns([1, 2])
    fi_name = fi_col.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index("Random Forest"), key="fi_model")
    fi = feature_importance(experiment.models[fi_name], FEATURE_COLUMNS)
    if fi is None:
        st.info("This model does not expose feature importance.")
    else:
        st.bar_chart(fi.set_index("feature")["importance"], height=230, color="#3B82F6")
        st.caption(
            "Trees/forests: impurity-based importance. Logistic regression: absolute standardised coefficient. "
            "Importance shows **association with the model output, not causation**. "
            "High importance for lecture counts is expected because the label is derived from them."
        )

# ══════════════════════════════════════════════════════════════════════════
# 5. Dataset & Methodology
# ══════════════════════════════════════════════════════════════════════════
with tab_meth:
    st.markdown("##### Dataset & Methodology")

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("""
        <div class="section-card">
        <div class="section-title">Data Provenance</div>
        The bundled <code>data/sample_attendance.csv</code> is <strong>synthetic</strong>, generated
        reproducibly by <code>src/data_utils.py</code> (seed 42). It contains <em>no real students</em>
        and no personal data. IDs like <code>STU0001</code> are entirely fictional.
        No public or licensed college attendance dataset was used.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card">
        <div class="section-title">Target — Approach A (Implemented)</div>
        <code>attendance_pct = lectures_attended / lectures_conducted &times; 100</code><br><br>
        <strong>At Risk</strong> if <code>attendance_pct &lt; threshold</code>
        (default {DEFAULT_THRESHOLD:.0f}% — an <em>example</em>; verify with your institution).<br>
        <strong>Not at Risk</strong> if at or above the threshold.<br><br>
        This is <strong>rule-based classification of the current situation</strong>, not a prediction of future risk.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
        <div class="section-title">Approach B (Not Implemented)</div>
        Future-risk prediction would use attendance history available <em>before</em> a cutoff date
        to predict a label observed <em>after</em> it. Requires real time-stamped data and is deliberately
        not mixed with Approach A.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="section-card">
        <div class="section-title">Target Leakage</div>
        The label is a deterministic function of <code>lectures_attended</code> and
        <code>lectures_conducted</code>. Those features contain the answer. The derived
        <code>attendance_pct</code> is therefore <strong>not</strong> given to the models as a feature —
        they must re-learn the ratio from raw counts. Results are framed as <em>reproducing the rule</em>,
        not as evidence of generalisation.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
        <div class="section-title">Models</div>
        <strong>Logistic Regression, Decision Tree, Random Forest</strong> — each a scikit-learn <code>Pipeline</code>.<br><br>
        &bull; Median imputation (fitted on training data only)<br>
        &bull; Standard scaling for Logistic Regression<br>
        &bull; Stratified 75/25 split, fixed seed 42<br>
        &bull; Rule baseline &amp; majority-class baseline reported
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-card">
        <div class="section-title">Known Limitations</div>
        Synthetic data &middot; Rule-generated labels &middot; Small sample (600 students) &middot;
        Single train/test split (no cross-validation) &middot; Optional features are simulated &middot;
        No fairness analysis &middot; Threshold may differ across courses or institutions
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:0.5rem;">Data Dictionary</div>', unsafe_allow_html=True)
    st.dataframe(DATA_DICTIONARY, hide_index=True, width='stretch')

    st.markdown("""
    <div style="background:#FFF7ED;border:1px solid #FED7AA;border-left:4px solid #F97316;
         border-radius:8px;padding:0.85rem 1rem;font-size:0.85rem;color:#7C2D12;margin-top:0.75rem;">
    <strong>&#x2696;&#xFE0F; Ethical Use.</strong> Treat outputs as a conversation starter, never as a verdict.
    Do not upload real student data to public deployments. No authentication, database,
    or external service is used by this application.
    </div>
    """, unsafe_allow_html=True)

