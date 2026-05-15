"""
streamlit_app.py - Interactive dashboard for UK house price forecasting.
Run with: streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_ingest import load_clean_data
from features import build_feature_matrix, get_feature_columns
from train import (train_test_split_temporal, train_xgboost, train_sarima,
                   train_baseline, predict_baseline, evaluate_model)
from predict import forecast_xgboost, forecast_sarima, build_forecast_dates

st.set_page_config(page_title="UK House Price Forecast", layout="wide")
st.title("UK House Price Forecast Dashboard")
st.markdown("Forecast UK average house prices using XGBoost, SARIMA, and a naive baseline.")

with st.sidebar:
    st.header("Settings")
    horizon = st.slider("Forecast horizon (months)", 3, 24, 12, 3)
    test_pct = st.slider("Test set size (%)", 5, 30, 20, 5)
    show_ci = st.checkbox("Show SARIMA confidence intervals", True)
    st.caption("Built by Swastika Dey | UCL")


@st.cache_data
def load_data():
    try:
        return load_clean_data()
    except Exception as e:
        st.error(f"Data load failed: {e}. Run src/data_ingest.py first.")
        return None


@st.cache_resource
def get_models(test_size):
    df = load_data()
    if df is None:
        return None
    feat_df = build_feature_matrix(df)
    cols = get_feature_columns(feat_df)
    train, test = train_test_split_temporal(feat_df, test_size / 100)
    xgb = train_xgboost(train[cols], train["avg_price"])
    sarima = train_sarima(train["avg_price"])
    baseline = train_baseline(train)
    return feat_df, cols, train, test, xgb, sarima, baseline


results = get_models(test_pct)
if results is None:
    st.stop()

feat_df, cols, train_df, test_df, xgb, sarima, baseline = results

# --- Test set evaluation ---
xgb_test = xgb.predict(test_df[cols])
sarima_test, _ = forecast_sarima(sarima, len(test_df))
base_test = predict_baseline(baseline, len(test_df))

st.subheader("Model Performance on Test Set")
c1, c2, c3 = st.columns(3)
for c, name, preds in zip([c1, c2, c3],
                           ["XGBoost", "SARIMA", "Baseline"],
                           [xgb_test, sarima_test.values, base_test]):
    m = evaluate_model(test_df["avg_price"].values, preds, name)
    c.metric(f"{name} MAE", f"GBP {m['mae']:,.0f}")
    c.metric(f"{name} MAPE", f"{m['mape']:.2f}%")

# --- Historical chart ---
st.subheader("Historical Prices + Test Set Predictions")
fig = go.Figure()
fig.add_trace(go.Scatter(x=feat_df["date"], y=feat_df["avg_price"],
                         name="Actual", line=dict(color="white")))
fig.add_trace(go.Scatter(x=test_df["date"], y=xgb_test,
                         name="XGBoost", line=dict(color="cyan", dash="dash")))
fig.add_trace(go.Scatter(x=test_df["date"], y=sarima_test.values,
                         name="SARIMA", line=dict(color="orange", dash="dot")))
fig.update_layout(template="plotly_dark", height=400,
                  xaxis_title="Date", yaxis_title="Price (GBP)")
st.plotly_chart(fig, use_container_width=True)

# --- Future forecast ---
st.subheader(f"Future Forecast: Next {horizon} Months")
last_date = feat_df["date"].max()
future_dates = build_forecast_dates(last_date, horizon)
xgb_future = forecast_xgboost(xgb, feat_df.iloc[-1], cols, horizon)
sarima_mean, sarima_ci = forecast_sarima(sarima, horizon)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=feat_df["date"].tail(36), y=feat_df["avg_price"].tail(36),
                           name="Historical", line=dict(color="white")))
fig2.add_trace(go.Scatter(x=future_dates, y=xgb_future,
                           name="XGBoost forecast", line=dict(color="cyan")))
fig2.add_trace(go.Scatter(x=future_dates, y=sarima_mean.values,
                           name="SARIMA forecast", line=dict(color="orange")))
if show_ci:
    upper = sarima_ci.iloc[:, 1].tolist()
    lower = sarima_ci.iloc[:, 0].tolist()
    fig2.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"
    ))
fig2.update_layout(template="plotly_dark", height=450,
                   xaxis_title="Date", yaxis_title="Price (GBP)")
st.plotly_chart(fig2, use_container_width=True)

# --- Forecast table ---
st.subheader("Forecast Table")
st.dataframe(pd.DataFrame({
    "Date": future_dates.strftime("%b %Y"),
    "XGBoost": [f"GBP {p:,.0f}" for p in xgb_future],
    "SARIMA": [f"GBP {p:,.0f}" for p in sarima_mean.values],
}), use_container_width=True)
