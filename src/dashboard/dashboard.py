"""
dashboard.py — FDA Biotech Stock Predictor Dashboard

Run with:
    streamlit run src/dashboard/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FDA Biotech Stock Predictor",
    page_icon="💊",
    layout="wide",
)


# ── Load data and models ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    master = pd.read_csv(
        "data/processed/master_dataset.csv",
        parse_dates=["approval_date"]
    )
    results = pd.read_csv(
        "data/processed/results.csv",
        parse_dates=["approval_date"]
    )
    return master, results


@st.cache_resource
def load_models():
    classifier = joblib.load("data/processed/classifier.joblib")
    regressor = joblib.load("data/processed/regressor.joblib")
    scaler = joblib.load("data/processed/scaler.joblib")
    return classifier, regressor, scaler


master, results = load_data()
classifier, regressor, scaler = load_models()

FEATURE_COLUMNS = [
    "phase_num", "enrollment", "is_priority_review",
    "volatility_30d", "approval_month", "approval_quarter",
    "filing_found", "trial_found",
]

FEATURE_DISPLAY_NAMES = {
    "phase_num": "Trial Phase",
    "enrollment": "Trial Enrollment Size",
    "is_priority_review": "Priority Review",
    "volatility_30d": "Stock Volatility (30d)",
    "approval_month": "Approval Month",
    "approval_quarter": "Approval Quarter",
    "filing_found": "SEC Filing Found",
    "trial_found": "Trial Data Found",
}


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("FDA Biotech Stock Predictor")
st.markdown("""
When the FDA approves or rejects a drug, the company's stock can move 
20 to 30% in a single day. This project builds a machine learning model 
that predicts which direction the stock will move in the 72 hours after 
the decision — using only publicly available data.
""")
st.divider()


# ── Section 1: The Data ────────────────────────────────────────────────────────
st.header("1. What Does the Data Look Like?")
st.markdown("""
We collected 65 branded drug approval events from 2010 to present, 
pulling data from four free government sources. Here is how stock prices 
actually moved after those decisions.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Events", len(master))
with col2:
    up_pct = (master["direction"] == 1).mean() * 100
    st.metric("Went UP", f"{up_pct:.0f}%")
with col3:
    down_pct = (master["direction"] == 0).mean() * 100
    st.metric("Went DOWN", f"{down_pct:.0f}%")
with col4:
    avg_move = master["pct_change_day3"].abs().mean()
    st.metric("Avg Absolute Move", f"{avg_move:.1f}%")

st.markdown("Most approvals cause small movements, but some cause huge ones. The model needs to learn to distinguish between them.")

fig1 = px.histogram(
    master,
    x="pct_change_day3",
    color=master["direction"].map({1: "UP", 0: "DOWN"}),
    color_discrete_map={"UP": "#22c55e", "DOWN": "#ef4444"},
    nbins=20,
    barmode="overlay",
    opacity=0.7,
    labels={
        "pct_change_day3": "3-Day Price Change (%)",
        "color": "Direction"
    },
    title="Distribution of Stock Price Movements After FDA Decisions"
)
fig1.update_layout(plot_bgcolor="white", margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig1, use_container_width=True)

st.divider()


# ── Section 2: What Drives the Movement? ──────────────────────────────────────
st.header("2. What Drives the Stock Movement?")
st.markdown("""
Before training the model, we can already see patterns in the data. 
Phase 3 trials — the final and largest stage before approval — tend to 
cause the biggest positive reactions because they carry the most 
credibility with investors.
""")

col1, col2 = st.columns(2)

with col1:
    phase_data = master[master["phase_num"] > 0].groupby("phase_num").agg(
        avg_change=("pct_change_day3", "mean"),
        count=("pct_change_day3", "count"),
    ).reset_index()
    phase_data["phase_label"] = "Phase " + phase_data["phase_num"].astype(str)

    fig2 = px.bar(
        phase_data,
        x="phase_label",
        y="avg_change",
        color="avg_change",
        color_continuous_scale=["#ef4444", "#f97316", "#22c55e"],
        text=phase_data["avg_change"].apply(lambda x: f"{x:+.1f}%"),
        labels={
            "avg_change": "Avg 3-Day Change (%)",
            "phase_label": "Trial Phase"
        },
        title="Average Price Impact by Trial Phase",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    priority_data = master.groupby("is_priority_review").agg(
        avg_change=("pct_change_day3", "mean"),
        count=("pct_change_day3", "count"),
    ).reset_index()
    priority_data["label"] = priority_data["is_priority_review"].map(
        {0: "Standard Review", 1: "Priority Review"}
    )

    fig3 = px.bar(
        priority_data,
        x="label",
        y="avg_change",
        color="avg_change",
        color_continuous_scale=["#ef4444", "#22c55e"],
        text=priority_data["avg_change"].apply(lambda x: f"{x:+.1f}%"),
        labels={"avg_change": "Avg 3-Day Change (%)", "label": "Review Type"},
        title="Priority vs Standard Review: Price Impact",
    )
    fig3.update_traces(textposition="outside")
    fig3.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()


# ── Section 3: Model Results ───────────────────────────────────────────────────
st.header("3. How Did the Model Perform?")
st.markdown("""
The model was trained on events from 2010 to 2022 and tested on events 
from 2023 onwards — data it had never seen. This is the honest way to 
evaluate a model because it simulates real-world use.
""")

accuracy = (results["predicted_direction"] == results["direction"]).mean() * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", f"{accuracy:.1f}%")
with col2:
    st.metric("Naive Baseline", "50.0%", help="What you would get by always guessing UP")
with col3:
    st.metric("Improvement", f"+{accuracy-50:.1f}%")

st.markdown("""
The chart below shows every prediction on the test set. Green dots are 
correct predictions, red dots are wrong ones. Points closer to the dashed 
line had more accurate magnitude predictions.
""")

fig4 = px.scatter(
    results,
    x="predicted_pct_change",
    y="pct_change_day3",
    color=results["correct"].map({1: "Correct", 0: "Wrong"}),
    color_discrete_map={"Correct": "#22c55e", "Wrong": "#ef4444"},
    hover_data=["ticker", "brand_name", "approval_date"],
    labels={
        "predicted_pct_change": "Predicted Change (%)",
        "pct_change_day3": "Actual Change (%)",
    },
    title="Predicted vs Actual Price Change (Test Set)",
)

min_val = min(results["predicted_pct_change"].min(), results["pct_change_day3"].min())
max_val = max(results["predicted_pct_change"].max(), results["pct_change_day3"].max())
fig4.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    line=dict(dash="dash", color="gray", width=1),
    name="Perfect prediction",
))
fig4.update_layout(plot_bgcolor="white", margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig4, use_container_width=True)

st.markdown("**What drove the predictions?** The model relied most heavily on trial enrollment size and stock volatility.")

importance_df = pd.DataFrame({
    "Feature": [FEATURE_DISPLAY_NAMES[f] for f in FEATURE_COLUMNS],
    "Importance": classifier.feature_importances_,
}).sort_values("Importance", ascending=True)

fig5 = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale=["#93c5fd", "#1d4ed8"],
    title="Which Features Drove the Model's Predictions",
)
fig5.update_layout(
    plot_bgcolor="white",
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig5, use_container_width=True)

st.divider()


# ── Section 4: Live Predictor ──────────────────────────────────────────────────
st.header("4. Try It Yourself")
st.markdown("""
Enter the details of a hypothetical FDA approval event below and the 
model will predict how the stock is likely to move. This uses the same 
trained model that produced the results above.
""")

col1, col2 = st.columns(2)

with col1:
    phase = st.selectbox(
        "Trial Phase",
        options=[1, 2, 3, 4],
        index=2,
        help="Phase 3 is the final stage before approval and tends to have the biggest market impact"
    )
    enrollment = st.slider(
        "Trial Enrollment (number of patients)",
        min_value=10,
        max_value=5000,
        value=500,
        step=10,
        help="Larger trials are generally seen as more credible by investors"
    )
    priority = st.radio(
        "Review Type",
        options=["Standard Review", "Priority Review"],
        help="Priority review means the FDA fast-tracked the drug for a serious condition"
    )

with col2:
    volatility = st.slider(
        "Stock Volatility (how much does the stock normally move day to day?)",
        min_value=0.5,
        max_value=8.0,
        value=2.0,
        step=0.1,
        help="Higher volatility means the stock swings more on average"
    )
    month = st.selectbox(
        "Approval Month",
        options=list(range(1, 13)),
        format_func=lambda x: [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ][x-1],
        index=5,
    )
    has_filing = st.radio(
        "Was an SEC filing found around this date?",
        options=["Yes", "No"],
    )

if st.button("Predict", type="primary"):
    quarter = (month - 1) // 3 + 1
    log_enrollment = np.log1p(enrollment)
    is_priority = 1 if priority == "Priority Review" else 0
    filing_found = 1 if has_filing == "Yes" else 0

    raw_features = pd.DataFrame([{
        "phase_num": phase,
        "enrollment": log_enrollment,
        "is_priority_review": is_priority,
        "volatility_30d": volatility,
        "approval_month": month,
        "approval_quarter": quarter,
        "filing_found": filing_found,
        "trial_found": 1,
    }])

    scaled_features = scaler.transform(raw_features)

    direction_pred = classifier.predict(scaled_features)[0]
    direction_proba = classifier.predict_proba(scaled_features)[0]
    magnitude_pred = regressor.predict(scaled_features)[0]

    confidence = direction_proba[direction_pred] * 100
    direction_label = "UP" if direction_pred == 1 else "DOWN"
    direction_color = "#22c55e" if direction_pred == 1 else "#ef4444"
    arrow = "↑" if direction_pred == 1 else "↓"

    st.markdown("---")
    st.subheader("Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Predicted Direction",
            f"{arrow} {direction_label}",
        )
    with col2:
        st.metric("Model Confidence", f"{confidence:.0f}%")
    with col3:
        st.metric("Predicted 3-Day Move", f"{magnitude_pred:+.1f}%")

    st.markdown("**Why did the model predict this?**")

    feature_importances = classifier.feature_importances_
    top_features = sorted(
        zip(FEATURE_COLUMNS, feature_importances),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    for feature, importance in top_features:
        feature_name = FEATURE_DISPLAY_NAMES[feature]
        pct = importance * 100
        st.markdown(f"- **{feature_name}** contributed {pct:.0f}% to this prediction")

st.divider()
st.caption("Built with real data from the FDA, SEC EDGAR, ClinicalTrials.gov, and Yahoo Finance.")