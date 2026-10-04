# pylint: disable=wrong-import-position
"""
dashboard/app.py
----------------
Streamlit dashboard for the Macro-Driven Credit Risk Lab.

Tabs:
  Overview          - forecasts with model-based intervals, history, stress summary, diagnostics
  Impulse responses - IRFs (with bands where available) and FEVD for the selected model
  Stress scenarios  - configured scenarios: macro paths, loss paths, cumulative losses vs. 2008
  Custom scenario   - build a shock and see its loss path against the baseline
  Backtests         - out-of-sample results for the loss model and the forecasting models
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.scenarios import HORIZON, SCENARIOS
from models.bvar import BVARModel
from models.loss_model import LossModel
from pipeline.run_pipeline import build_model, load_data, run_pipeline
from pipeline.scenario_engine import (REFERENCE_2008, VARS, build_path, cumulative_loss,
                                      load_history, loss_path, run_scenarios, scenario_summary)

FIG_DIR = ROOT / "reports" / "figures"
MODEL_LABELS = {
    "bvar": "BVAR (Minnesota prior)",
    "var": "VAR (reduced form)",
    "svar_cholesky": "SVAR (Cholesky)",
    "svar_sign": "SVAR (sign restrictions)",
    "local_projections": "Local projections",
}

st.set_page_config(page_title="Macro Credit Risk Lab", layout="wide")


# ── Cached computations ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running forecast pipeline...")
def cached_pipeline(model_type: str, lags: int, alpha: float) -> dict:
    return run_pipeline(model_type=model_type, var_lags=lags, alpha=alpha, run_stress=False)


@st.cache_resource(show_spinner="Fitting model...")
def cached_model(model_type: str, lags: int):
    return build_model(model_type, lags, load_data())


@st.cache_data(show_spinner="Running stress scenarios...")
def cached_scenarios():
    results, history = run_scenarios()
    return results, history, scenario_summary(results)


@st.cache_resource
def scenario_models():
    history = load_history()
    return history, BVARModel(lags=2, lambda_=0.2).fit(history[VARS]), LossModel.load()


def path_chart(paths: dict, height: int = 360) -> go.Figure:
    """Unemployment, inflation, and charge-off paths for several named scenarios."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=["Unemployment (%)", "Inflation (%)",
                                                         "Charge-off rate (%)"])
    for name, path in paths.items():
        for col, (series, scale) in enumerate([("unemployment", 1), ("inflation", 1),
                                               ("loss_rate", 100)], start=1):
            fig.add_trace(go.Scatter(x=path.index, y=path[series] * scale, mode="lines",
                                     name=name, legendgroup=name, showlegend=(col == 1)),
                          row=1, col=col)
    fig.update_layout(height=height, legend=dict(orientation="h", y=-0.2))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")
model_type = st.sidebar.selectbox("Forecasting model", list(MODEL_LABELS),
                                  format_func=MODEL_LABELS.get)
lags = st.sidebar.slider("Lag order", 1, 4, 2)
alpha = st.sidebar.select_slider("Interval level", options=[0.20, 0.10, 0.05], value=0.10,
                                 format_func=lambda a: f"{1 - a:.0%}")
st.sidebar.caption("Stress scenarios always use BVAR(2), the most robust model "
                   "in the out-of-sample forecast backtest.")

result = cached_pipeline(model_type, lags, alpha)
model = cached_model(model_type, lags)
results, history, summary = cached_scenarios()
ref = history.loc[REFERENCE_2008[0]:REFERENCE_2008[1], "charge_off_rate"] / 100
ref_cumulative = cumulative_loss(ref) * 100

tab_overview, tab_irf, tab_stress, tab_custom, tab_backtest = st.tabs(
    ["Overview", "Impulse responses", "Stress scenarios", "Custom scenario", "Backtests"])

# ── Overview ──────────────────────────────────────────────────────────────────
with tab_overview:
    st.title("Macro Credit Risk Lab")
    st.caption("Bayesian and structural VARs, local projections, a dynamic credit loss model, "
               "and model-based stress testing on FRED data, 1985 to present.")

    fc, ci, level = result["forecast"], result["forecast_ci"], f"{1 - alpha:.0%}"
    latest_co = history["charge_off_rate"].dropna()
    base_cum = summary.set_index("scenario").loc["Baseline", "cumulative_9q_loss_pct"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unemployment, next quarter", f"{fc['unemployment']:.2f}%")
    c1.caption(f"{level} interval: {ci['unemployment']['lower']:.2f}% to {ci['unemployment']['upper']:.2f}%")
    c2.metric("Inflation, next quarter", f"{fc['inflation']:.2f}%")
    c2.caption(f"{level} interval: {ci['inflation']['lower']:.2f}% to {ci['inflation']['upper']:.2f}%")
    c3.metric("Latest charge-off rate", f"{latest_co.iloc[-1]:.2f}%")
    c3.caption(f"Consumer loans, {latest_co.index[-1]:%Y} Q{latest_co.index[-1].quarter}")
    c4.metric("Baseline 9-quarter loss", f"{base_cum:.2f}%")
    c4.caption(f"Model: {MODEL_LABELS[model_type]}, {lags} lags")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=["Unemployment (%)", "Inflation, year over year (%)",
                                        "Consumer charge-off rate (%)"])
    for row, col in enumerate(["unemployment", "inflation", "charge_off_rate"], start=1):
        fig.add_trace(go.Scatter(x=history.index, y=history[col], mode="lines",
                                 showlegend=False), row=row, col=1)
    fig.update_layout(height=600, margin=dict(t=40, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Stress test summary")
    st.dataframe(summary.round(2), width="stretch", hide_index=True)
    st.caption(f"Reference: actual cumulative loss, 2008 Q1 to 2010 Q1, was {ref_cumulative:.2f}%.")

    st.subheader("Diagnostics")
    stat = pd.DataFrame([{"Variable": k.capitalize(), "ADF statistic": v["adf_stat"],
                          "p-value": v["p_value"], "Stationary (5%)": v["stationary"]}
                         for k, v in result["stationarity"].items()])
    st.dataframe(stat, width="stretch", hide_index=True)
    notes = [f"Lags: {result.get('var_lags_used')}"]
    if result.get("var_stability") is not None:
        notes.append(f"Stable: {result['var_stability']}")
    if result.get("sign_acceptance_rate") is not None:
        notes.append(f"Sign-restriction acceptance rate: {result['sign_acceptance_rate']:.0%}")
    st.caption(" | ".join(notes))

# ── Impulse responses ─────────────────────────────────────────────────────────
with tab_irf:
    st.title("Impulse responses")
    bundle = model.irf()
    irfs = np.array(bundle.irfs)
    bands = None
    if model_type == "svar_sign":
        bands = model.irf_bands()
        st.caption("Median across accepted rotations, with 16th-84th percentile bands. A demand-type "
                   "unemployment shock raises unemployment and lowers inflation for three quarters.")
    elif model_type == "local_projections":
        bands = model.get_ci(alpha)
        st.caption(f"Local projection responses with {level} HAC confidence bands.")
    elif model_type == "var":
        st.caption("Reduced-form responses to one-unit innovations (not structural).")
    else:
        st.caption("Cholesky-identified responses: unemployment ordered first.")

    shock = st.selectbox("Shock", VARS, format_func=lambda v: f"{v.capitalize()} shock")
    j = VARS.index(shock)
    h = np.arange(len(irfs))
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{v.capitalize()} response" for v in VARS])
    for i in range(len(VARS)):
        if bands is not None:
            lo, hi = bands
            n = min(len(h), len(lo))
            fig.add_trace(go.Scatter(x=np.concatenate([h[:n], h[:n][::-1]]),
                                     y=np.concatenate([hi[:n, i, j], lo[:n, i, j][::-1]]),
                                     fill="toself", line=dict(width=0), opacity=0.25,
                                     showlegend=False, hoverinfo="skip"), row=1, col=i + 1)
        fig.add_trace(go.Scatter(x=h, y=irfs[:, i, j], mode="lines+markers", showlegend=False),
                      row=1, col=i + 1)
        fig.add_hline(y=0, line_dash="dash", line_color="grey", row=1, col=i + 1)
    fig.update_xaxes(title_text="Quarters after shock")
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

    if bundle.fevd is not None:
        st.subheader("Forecast error variance decomposition")
        fevd = np.array(bundle.fevd)                       # (horizon, variable, shock)
        resp = st.selectbox("Variable", VARS, format_func=str.capitalize, key="fevd_var")
        r = VARS.index(resp)
        fig = go.Figure([go.Bar(x=np.arange(fevd.shape[0]), y=fevd[:, r, k] * 100,
                                name=f"{VARS[k].capitalize()} shock") for k in range(len(VARS))])
        fig.update_layout(barmode="stack", height=340, yaxis_title="Share of variance (%)",
                          xaxis_title="Horizon (quarters)")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Local projections estimate responses directly and do not produce an FEVD.")

# ── Stress scenarios ──────────────────────────────────────────────────────────
with tab_stress:
    st.title("Stress scenarios")
    st.caption("Structural shocks propagated through a BVAR(2), feeding the dynamic loss model. "
               "No pandemic-style policy support is assumed.")
    st.dataframe(pd.DataFrame([{"Scenario": k, "Description": v["description"]}
                               for k, v in SCENARIOS.items()]),
                 width="stretch", hide_index=True)

    chosen = st.multiselect("Scenarios to compare", list(SCENARIOS), default=list(SCENARIOS))
    if chosen:
        st.plotly_chart(path_chart({k: results[k] for k in chosen}), width="stretch")

        sub = summary[summary["scenario"].isin(chosen)]
        fig = go.Figure(go.Bar(x=sub["scenario"], y=sub["cumulative_9q_loss_pct"],
                               text=sub["cumulative_9q_loss_pct"].round(2), textposition="outside"))
        fig.add_hline(y=ref_cumulative, line_dash="dash", line_color="red",
                      annotation_text=f"Actual 2008-2010: {ref_cumulative:.2f}%")
        fig.update_layout(height=380, yaxis_title="Cumulative 9-quarter loss (%)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(sub.round(2), width="stretch", hide_index=True)

# ── Custom scenario ───────────────────────────────────────────────────────────
with tab_custom:
    st.title("Custom scenario")
    st.caption("Size each shock as a peak deviation from the baseline forecast, built up over a "
               "number of quarters. Paths follow the BVAR's estimated dynamics.")
    hist_s, bvar, loss_model = scenario_models()

    left, right = st.columns(2)
    u_peak = left.slider("Unemployment shock: peak (pp above baseline)", 0.0, 10.0, 4.0, 0.5)
    u_q = left.slider("Unemployment shock: quarters to build", 1, 8, 4)
    pi_peak = right.slider("Inflation shock: peak (pp above baseline)", 0.0, 8.0, 0.0, 0.5)
    pi_q = right.slider("Inflation shock: quarters to build", 1, 8, 4)

    shocks = {}
    if u_peak > 0:
        shocks["unemployment"] = {"peak": u_peak, "quarters": u_q}
    if pi_peak > 0:
        shocks["inflation"] = {"peak": pi_peak, "quarters": pi_q}

    custom = build_path(bvar, hist_s.index[-1], HORIZON, shocks)
    custom["loss_rate"] = loss_path(loss_model, hist_s, custom)
    baseline = results["Baseline"]
    custom_cum = cumulative_loss(custom["loss_rate"]) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Peak unemployment", f"{custom['unemployment'].max():.2f}%")
    m2.metric("Peak charge-off rate", f"{custom['loss_rate'].max() * 100:.2f}%")
    m3.metric("Cumulative 9-quarter loss", f"{custom_cum:.2f}%",
              f"{custom_cum - base_cum:+.2f} pp vs. baseline", delta_color="inverse")
    st.plotly_chart(path_chart({"Baseline": baseline, "Custom": custom}),
                    width="stretch")

# ── Backtests ─────────────────────────────────────────────────────────────────
with tab_backtest:
    st.title("Backtests")

    def show_image(name: str, caption: str):
        p = FIG_DIR / name
        if p.exists():
            st.image(str(p), caption=caption, width="stretch")
        else:
            st.info(f"Run `make all` to generate {name}.")

    st.subheader("Dynamic credit loss model")
    show_image("loss_model_backtests.png",
               "Conditional backtests: losses simulated from the actual macro path, "
               "using a model trained through 2006.")
    st.subheader("Static levels model (for comparison)")
    show_image("credit_link_oot.png",
               "A static levels model fails out of time: it misses 2008 by about 3 points.")

    st.subheader("Forecasts vs. a random walk (2000 onward)")
    for file, label in [("forecast_backtest_ex_covid.csv", "Excluding COVID targets"),
                        ("forecast_backtest_full.csv", "Full sample, including COVID")]:
        p = FIG_DIR / file
        if p.exists():
            st.markdown(f"**{label}** (relative RMSE below 1 beats the random walk)")
            st.dataframe(pd.read_csv(p).round(3), width="stretch", hide_index=True)
    show_image("forecast_backtest.png", "Four-quarter-ahead forecasts vs. actual outcomes.")
