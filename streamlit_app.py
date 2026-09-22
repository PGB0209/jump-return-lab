"""
streamlit_app.py
------------------
A local, live web app version of the jump-return toolkit. Run it with:

    streamlit run streamlit_app.py

This opens a real webpage in your browser (usually http://localhost:8501)
that fetches LIVE data from Yahoo Finance for whatever ticker you type in,
runs the full jump-detection / statistical-test / walk-forward-backtest
pipeline, and renders interactive charts — all reusing the exact same
analysis code (data_loader.py, jump_detector.py, etc.) as the CLI tool.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_from_yahoo, generate_synthetic
from jump_detector import detect_jumps, jump_summary
from forward_returns import analyze_jump_outcomes, compute_forward_returns, interpret_results
from features import build_features, FEATURE_COLUMNS
from backtest import walk_forward_backtest, summarize_backtest

st.set_page_config(page_title="Jump Return Lab", page_icon="📈", layout="wide")

# ---------- Sidebar controls ----------
st.sidebar.title("📈 Jump Return Lab")
st.sidebar.caption("Live jump-based return analysis")

data_source = st.sidebar.radio("Data source", ["Live ticker (Yahoo Finance)", "Synthetic (no network)"])

if data_source == "Live ticker (Yahoo Finance)":
    ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
    start_date = st.sidebar.date_input("Start date", value=pd.Timestamp("2018-01-01"))
else:
    ticker = "SYNTHETIC"
    start_date = None

z_threshold = st.sidebar.slider("Jump z-score threshold", 1.5, 4.0, 2.5, 0.1)
vol_window = st.sidebar.slider("Volatility window (days)", 10, 60, 20, 5)
horizon = st.sidebar.select_slider("Backtest horizon (days)", options=[1, 3, 5, 10, 20], value=5)
model_type = st.sidebar.selectbox("Backtest model", ["logistic", "xgboost"])
run_button = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Not investment advice. This tests whether a jump-based signal shows "
    "historical statistical significance — many honest results will show none."
)

# ---------- Main ----------
st.title(f"Jump Analysis: {ticker}" if ticker else "Jump Return Lab")

if not run_button and "last_run" not in st.session_state:
    st.info("Set your parameters in the sidebar and click **Run analysis** to begin.")
    st.stop()

if run_button:
    st.session_state["last_run"] = True

with st.spinner(f"Loading data for {ticker}..."):
    try:
        if data_source == "Live ticker (Yahoo Finance)":
            df = load_from_yahoo(ticker, start=str(start_date))
        else:
            df = generate_synthetic()
    except Exception as e:
        st.error(f"Couldn't load data: {e}")
        st.stop()

if len(df) < 200:
    st.warning(f"Only {len(df)} rows of data loaded — results may be unreliable with such a short history.")

with st.spinner("Detecting jumps and running statistics..."):
    df = detect_jumps(df, vol_window=vol_window, z_threshold=z_threshold)
    summary = jump_summary(df)
    fwd_results = analyze_jump_outcomes(df)
    fwd = compute_forward_returns(df, horizons=(1, 3, 5, 10, 20))

# ---------- Summary stats row ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jumps detected", summary["num_jumps"])
c2.metric("Jump rate", f"{summary['jump_rate_pct']}%")
c3.metric("Up / Down", f"{summary['num_up_jumps']} / {summary['num_down_jumps']}")
c4.metric("Avg up jump", f"{summary['avg_up_jump_return_pct']}%" if summary["avg_up_jump_return_pct"] is not None else "—")
c5.metric("Avg down jump", f"{summary['avg_down_jump_return_pct']}%" if summary["avg_down_jump_return_pct"] is not None else "—")

# ---------- Price chart ----------
st.subheader("Price with detected jumps")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close",
                          line=dict(color="#3ECFB5", width=1.3)))
up = df[df["JumpDirection"] == 1]
down = df[df["JumpDirection"] == -1]
fig.add_trace(go.Scatter(x=up.index, y=up["Close"], mode="markers", name="Up jump",
                          marker=dict(color="#34D399", size=9, symbol="triangle-up")))
fig.add_trace(go.Scatter(x=down.index, y=down["Close"], mode="markers", name="Down jump",
                          marker=dict(color="#F1716B", size=9, symbol="triangle-down")))
fig.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# ---------- Forward return distribution + stats table ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"{horizon}-day forward return distribution")
    fwd_col = f"Fwd_{horizon}"
    baseline = fwd.loc[~fwd["IsJump"], fwd_col].dropna() * 100
    up_d = fwd.loc[fwd["JumpDirection"] == 1, fwd_col].dropna() * 100
    down_d = fwd.loc[fwd["JumpDirection"] == -1, fwd_col].dropna() * 100

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=baseline, name=f"Non-jump (n={len(baseline)})",
                                     opacity=0.5, histnorm="probability density", marker_color="#8792AC"))
    if len(up_d) >= 5:
        hist_fig.add_trace(go.Histogram(x=up_d, name=f"After up jump (n={len(up_d)})",
                                         opacity=0.6, histnorm="probability density", marker_color="#34D399"))
    if len(down_d) >= 5:
        hist_fig.add_trace(go.Histogram(x=down_d, name=f"After down jump (n={len(down_d)})",
                                         opacity=0.6, histnorm="probability density", marker_color="#F1716B"))
    hist_fig.update_layout(barmode="overlay", height=360, template="plotly_dark",
                            margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(hist_fig, use_container_width=True)

with col2:
    st.subheader("Statistical test results")
    st.dataframe(fwd_results, use_container_width=True, height=360)

st.info(interpret_results(fwd_results))

# ---------- Walk-forward backtest ----------
st.subheader(f"Walk-forward backtest ({model_type}, {horizon}-day horizon)")
with st.spinner("Running walk-forward backtest..."):
    df_feat = build_features(df)
    try:
        train_w = min(400, len(df_feat) // 3)
        test_w = max(20, len(df_feat) // 15)
        bt = walk_forward_backtest(df_feat, FEATURE_COLUMNS, horizon=horizon,
                                    train_window=train_w, test_window=test_w, model_type=model_type)
        bt_summary = summarize_backtest(bt)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Model accuracy", f"{bt_summary['avg_model_accuracy']*100:.1f}%")
        b2.metric("Naive 'always up' baseline", f"{bt_summary['avg_naive_up_accuracy']*100:.1f}%")
        b3.metric("Beats baseline in", f"{bt_summary['model_beats_naive_pct_of_folds']}% of folds")
        b4.metric("Backtest folds", bt_summary["num_folds"])

        bt_fig = go.Figure()
        bt_fig.add_trace(go.Scatter(x=bt["test_start"], y=bt["model_accuracy"], mode="lines+markers",
                                     name="Model", line=dict(color="#3ECFB5")))
        bt_fig.add_trace(go.Scatter(x=bt["test_start"], y=bt["naive_up_accuracy"], mode="lines+markers",
                                     name="Naive 'always up'", line=dict(color="#8792AC", dash="dash")))
        bt_fig.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="coin flip")
        bt_fig.update_layout(height=340, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(bt_fig, use_container_width=True)

        with st.expander("Full per-fold backtest results"):
            st.dataframe(bt, use_container_width=True)

        if bt_summary["model_beats_naive_pct_of_folds"] < 55:
            st.warning(
                "The model beats the naive baseline in fewer than 55% of folds — treat this as "
                "**no reliable edge**, not a working signal. Transaction costs aren't modeled here either, "
                "which would erode any small edge further."
            )
    except Exception as e:
        st.error(f"Backtest couldn't be run: {e}")

st.divider()
st.caption(
    "Research/educational tool, not investment advice. Past patterns are not guarantees of future returns."
)
