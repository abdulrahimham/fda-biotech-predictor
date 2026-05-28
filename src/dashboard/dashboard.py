"""
dashboard.py — Interactive web dashboard for the FDA Biotech Stock Predictor.

Run with:
    streamlit run src/dashboard/dashboard.py

This is what you show in interviews. It turns all our data and model
results into a visual story that anyone can understand.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib


# Page config 
st.set_page_config(
    page_title="FDA Biotech Stock Predictor",
    layout="wide",
)


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    master = pd.read_csv("data/processed/master_dataset.csv", parse_dates=["approval_date"])
    results = pd.read_csv("data/processed/results.csv", parse_dates=["approval_date"])
    return master, results


@st.cache_resource
def load_models():
    classifier = joblib.load("data/processed/classifier.joblib")
    regressor = joblib.load("data/processed/regressor.joblib")
    return classifier, regressor


master, results = load_data()
classifier, regressor = load_models()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("FDA Biotech Stock Predictor")
st.markdown("*Predicting stock price movements around FDA drug approval events*")
st.divider()


# ── Top metrics ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Events", len(master))

with col2:
    up_pct = (master["direction"] == 1).mean() * 100
    st.metric("Stocks Went UP", f"{up_pct:.0f}%")

with col3:
    avg_move = master["pct_change_day3"].mean()
    st.metric("Avg 3-Day Move", f"{avg_move:+.1f}%")

with col4:
    accuracy = (results["predicted_direction"] == results["direction"]).mean() * 100
    st.metric("Model Accuracy", f"{accuracy:.1f}%", delta="vs 50% baseline")

st.divider()


# ── Row 1: Price distribution + Feature importance ─────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Movement Distribution")

    fig = px.histogram(
        master,
        x="pct_change_day3",
        color=master["direction"].map({1: "UP", 0: "DOWN"}),
        color_discrete_map={"UP": "#22c55e", "DOWN": "#ef4444"},
        nbins=20,
        barmode="overlay",
        opacity=0.7,
        labels={"pct_change_day3": "3-Day Price Change (%)", "color": "Direction"},
    )
    fig.update_layout(
        plot_bgcolor="white",
        legend_title_text="Direction",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("What Drove Predictions")

    # Feature importance from the classifier
    feature_names = [
        "phase_num", "enrollment", "is_priority_review",
        "volatility_30d", "approval_month", "approval_quarter",
        "filing_found", "trial_found",
    ]
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_,
    }).sort_values("Importance", ascending=True)

    # Clean up feature names for display
    name_map = {
        "phase_num": "Trial Phase",
        "enrollment": "Trial Enrollment Size",
        "is_priority_review": "Priority Review",
        "volatility_30d": "Stock Volatility (30d)",
        "approval_month": "Approval Month",
        "approval_quarter": "Approval Quarter",
        "filing_found": "SEC Filing Found",
        "trial_found": "Trial Data Found",
    }
    importance_df["Feature"] = importance_df["Feature"].map(name_map)

    fig2 = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#93c5fd", "#1d4ed8"],
    )
    fig2.update_layout(
        plot_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ── Row 2: Predicted vs Actual ─────────────────────────────────────────────────
st.subheader("Predicted vs Actual Price Change (Test Set)")

fig3 = px.scatter(
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
    size_max=12,
)

# Add perfect prediction line
min_val = min(results["predicted_pct_change"].min(), results["pct_change_day3"].min())
max_val = max(results["predicted_pct_change"].max(), results["pct_change_day3"].max())
fig3.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    line=dict(dash="dash", color="gray", width=1),
    name="Perfect prediction",
))

fig3.update_layout(
    plot_bgcolor="white",
    margin=dict(l=0, r=0, t=20, b=0),
)
st.plotly_chart(fig3, use_container_width=True)


# ── Row 3: Results table ───────────────────────────────────────────────────────
st.subheader("Test Set Predictions")

display = results[[
    "ticker", "brand_name", "approval_date",
    "pct_change_day3", "predicted_pct_change",
    "direction", "predicted_direction", "correct"
]].copy()

display["approval_date"] = display["approval_date"].dt.strftime("%Y-%m-%d")
display["pct_change_day3"] = display["pct_change_day3"].apply(lambda x: f"{x:+.2f}%")
display["predicted_pct_change"] = display["predicted_pct_change"].apply(lambda x: f"{x:+.2f}%")
display["direction"] = display["direction"].map({1: "UP", 0: "DOWN"})
display["predicted_direction"] = display["predicted_direction"].map({1: "UP", 0: "DOWN"})
display["correct"] = display["correct"].map({1: "✓", 0: "✗"})

display.columns = [
    "Ticker", "Drug", "Date",
    "Actual Change", "Predicted Change",
    "Actual", "Predicted", "Correct"
]

st.dataframe(display, use_container_width=True, hide_index=True)


# ── Row 4: Price impact by phase ───────────────────────────────────────────────
st.subheader("Average Price Impact by Trial Phase")

phase_data = master[master["phase_num"] > 0].groupby("phase_num").agg(
    avg_change=("pct_change_day3", "mean"),
    count=("pct_change_day3", "count"),
).reset_index()
phase_data["phase_label"] = "Phase " + phase_data["phase_num"].astype(str)

fig4 = px.bar(
    phase_data,
    x="phase_label",
    y="avg_change",
    color="avg_change",
    color_continuous_scale=["#ef4444", "#f97316", "#22c55e"],
    text=phase_data["avg_change"].apply(lambda x: f"{x:+.1f}%"),
    labels={"avg_change": "Avg 3-Day Change (%)", "phase_label": "Trial Phase"},
)
fig4.update_traces(textposition="outside")
fig4.update_layout(
    plot_bgcolor="white",
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, t=20, b=0),
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption("Built with real FDA, SEC EDGAR, ClinicalTrials.gov, and Yahoo Finance data.")