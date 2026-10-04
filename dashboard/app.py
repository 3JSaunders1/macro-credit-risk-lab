"""
dashboard/app.py
----------------
Streamlit dashboard: interactive macro credit risk lab.

Features:
  - Model selection: VAR / SVAR-Cholesky / SVAR-Sign / BVAR / LP
  - IRF + FEVD visualization
  - PD gauge + stress test
  - Scenario comparison (2-5 worlds side by side)
  - Shock path engine (dynamic PD trajectory)
  - Run history + replay
  - Sensitivity heatmap
"""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pipeline.run_pipeline import (
    run_pipeline, run_scenario_comparison,
    build_shock_path, compute_pd_path, NAMED_SCENARIOS, load_data,
)
from models.credit_model import CreditModel

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Credit Risk Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#1f3a5f",
    "accent":  "#2e86ab",
    "warning": "#e07a5f",
    "green":   "#3d9970",
    "grey":    "#8d99ae",
}

RATING_COLOR = {
    "AAA": "#3d9970", "AA": "#3d9970", "A": "#3d9970",
    "BBB": "#f9c74f", "BB": "#f9c74f",
    "B": "#e07a5f", "CCC": "#e07a5f", "D": "#c1121f",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

tab_selected = st.sidebar.radio(
    "View",
    ["🏠 Overview", "📈 IRF / FEVD", "⚡ Shock Engine",
     "🌍 Scenario Comparison", "🕘 Run History"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")

model_type = st.sidebar.selectbox(
    "Model Type",
    ["var", "svar_cholesky", "svar_sign", "bvar", "local_projections"],
    format_func=lambda x: {
        "var": "VAR",
        "svar_cholesky": "SVAR (Cholesky)",
        "svar_sign": "SVAR (Sign Restrictions)",
        "bvar": "BVAR (Minnesota Prior)",
        "local_projections": "Local Projections",
    }[x],
)

var_lags = st.sidebar.slider("Lag Order", 1, 8, 2)
alpha = st.sidebar.slider("CI Level (α)", 0.05, 0.30, 0.10, step=0.05)
shock_size = st.sidebar.slider("Sensitivity Shock (pp)", 0.5, 3.0, 1.0, step=0.5)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶ Run Pipeline", use_container_width=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None

params_key = (model_type, var_lags, alpha, shock_size)
if "last_params" not in st.session_state:
    st.session_state.last_params = None

# Auto-run or manual run
if run_btn or st.session_state.last_params != params_key:
    st.session_state.last_params = params_key
    with st.spinner("Running pipeline..."):
        try:
            result = run_pipeline(
                model_type=model_type,
                var_lags=var_lags,
                alpha=alpha,
                shock=shock_size,
                run_stress=True,
            )
            st.session_state.result = result
            st.session_state.run_history.append({
                "id": result["experiment_id"],
                "model": model_type,
                "lags": var_lags,
                "pd": result["predicted_pd"],
                "rating": result["rating"],
                "u_forecast": result["forecast"]["unemployment"],
                "pi_forecast": result["forecast"]["inflation"],
            })
        except Exception as e:
            st.error(f"Pipeline error: {e}")

result = st.session_state.result

# ── Helper: guard ──────────────────────────────────────────────────────────────
def require_result():
    if result is None:
        st.info("Click **▶ Run Pipeline** to get started.")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TAB: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if tab_selected == "🏠 Overview":
    st.title("📊 Macro Credit Risk Lab")
    st.caption("VAR · SVAR · BVAR · Local Projections · Credit Risk · Stress Testing")

    require_result()

    # KPI row
    fc = result["forecast"]
    ci = result["forecast_ci"]
    pd_val = result["predicted_pd"]
    rating = result["rating"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Unemployment Forecast", f"{fc['unemployment']:.2f}%",
              f"CI: [{ci['unemployment']['lower']:.1f}, {ci['unemployment']['upper']:.1f}]")
    c2.metric("Inflation Forecast", f"{fc['inflation']:.2f}%",
              f"CI: [{ci['inflation']['lower']:.1f}, {ci['inflation']['upper']:.1f}]")
    c3.metric("Probability of Default", f"{pd_val:.2%}")
    c4.metric("Internal Rating", rating)
    c5.metric("Model", result["model_type"].upper().replace("_", " "))

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # Macro history
    with col_left:
        st.subheader("Macro History")
        df = load_data()
        df_plot = pd.read_csv(ROOT / "data/macro_data.csv", parse_dates=["date"])
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Unemployment (%)", "Inflation (%)"],
                            vertical_spacing=0.08)
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["unemployment"],
                                 line=dict(color=COLORS["primary"], width=2),
                                 name="Unemployment"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot["date"], y=df_plot["inflation"],
                                 line=dict(color=COLORS["accent"], width=2),
                                 name="Inflation"), row=2, col=1)
        fig.update_layout(height=350, showlegend=False, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # PD Gauge
    with col_right:
        st.subheader("PD Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_val * 100,
            number=dict(suffix="%", valueformat=".2f"),
            title=dict(text=f"PD — Rating: {rating}"),
            gauge=dict(
                axis=dict(range=[0, 25], ticksuffix="%"),
                bar=dict(color=RATING_COLOR.get(rating, COLORS["grey"])),
                steps=[
                    dict(range=[0, 2], color="#e8f5e9"),
                    dict(range=[2, 5], color="#fff9c4"),
                    dict(range=[5, 15], color="#ffe0b2"),
                    dict(range=[15, 25], color="#ffcdd2"),
                ],
            ),
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Sensitivity
        sens = result["sensitivity"]
        st.caption("Sensitivity to 1pp macro shocks")
        s1, s2 = st.columns(2)
        s1.metric("Δ Unemployment +1pp", f"{sens['unemployment_shock_delta']:+.3%}")
        s2.metric("Δ Inflation +1pp", f"{sens['inflation_shock_delta']:+.3%}")

    # Stationarity
    st.markdown("---")
    st.subheader("Stationarity (ADF Tests)")
    stat = result.get("stationarity", {})
    if stat:
        stat_df = pd.DataFrame([
            {
                "Variable": k.capitalize(),
                "ADF Stat": f"{v['adf_stat']:.4f}",
                "p-value": f"{v['p_value']:.4f}",
                "Stationary": "✅ Yes" if v["stationary"] else "❌ No",
            }
            for k, v in stat.items()
        ])
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

    # Stress test
    if result.get("stress_test") is not None:
        st.markdown("---")
        st.subheader("Stress Test")
        st.dataframe(result["stress_test"].round(2), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: IRF / FEVD
# ══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "📈 IRF / FEVD":
    st.title("📈 Impulse Response Functions & FEVD")
    require_result()

    irf_data = result.get("irf", {})
    irfs = irf_data.get("irfs", [])
    fevd = irf_data.get("fevd", None)

    if not irfs:
        st.warning("No IRF data available for this model/run.")
        st.stop()

    variables = ["Unemployment", "Inflation"]
    horizon = list(range(len(irfs)))

    shock_var = st.selectbox("Shock Variable", variables)
    shock_idx = variables.index(shock_var)

    st.subheader(f"Responses to a {shock_var} Shock")

    fig_irf = make_subplots(rows=1, cols=2,
                             subplot_titles=[f"{v} Response" for v in variables])

    for resp_idx, resp_var in enumerate(variables):
        vals = [float(irfs[h][resp_idx, shock_idx]) for h in horizon]
        fig_irf.add_trace(
            go.Scatter(x=horizon, y=vals,
                       mode="lines+markers",
                       line=dict(color=COLORS["primary"] if resp_idx == 0 else COLORS["accent"],
                                 width=2.5),
                       name=resp_var),
            row=1, col=resp_idx + 1
        )
        fig_irf.add_hline(y=0, line_dash="dash", line_color="grey",
                           opacity=0.5, row=1, col=resp_idx + 1)

    fig_irf.update_layout(height=380, showlegend=False,
                           xaxis_title="Horizon (quarters)",
                           xaxis2_title="Horizon (quarters)")
    st.plotly_chart(fig_irf, use_container_width=True)

    # FEVD
    if fevd is not None:
        st.markdown("---")
        st.subheader("Forecast Error Variance Decomposition (FEVD)")

        resp_var = st.selectbox("Response Variable", variables, key="fevd_var")
        resp_idx = variables.index(resp_var)

        fevd_arr = np.array(fevd)
        if fevd_arr.ndim == 3 and fevd_arr.shape[1] >= 2:
            h_range = list(range(fevd_arr.shape[0]))
            fig_fevd = go.Figure()
            for shock_i, shock_name in enumerate(variables):
                vals = [float(fevd_arr[h, resp_idx, shock_i]) * 100
                        for h in h_range]
                fig_fevd.add_trace(go.Bar(
                    x=h_range, y=vals, name=f"{shock_name} shock",
                ))
            fig_fevd.update_layout(
                barmode="stack", height=350,
                yaxis_title="Share of Variance (%)",
                xaxis_title="Horizon (quarters)",
            )
            st.plotly_chart(fig_fevd, use_container_width=True)
    else:
        if result["model_type"] == "local_projections":
            st.info("FEVD is not computed for Local Projections (LP-IRF). "
                    "Use VAR or SVAR for FEVD.")

    # Model info
    st.markdown("---")
    st.caption(f"Model: `{result['model_type']}` | "
               f"Lags: {result.get('var_lags_used')} | "
               f"Stable: {result.get('var_stability')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: SHOCK ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "⚡ Shock Engine":
    st.title("⚡ Dynamic Shock Engine")
    st.caption("Define a macro shock and see PD evolve over the horizon as it decays.")

    require_result()

    fc = result["forecast"]
    base_u = fc["unemployment"]
    base_pi = fc["inflation"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Shock Parameters")
        shock_u = st.slider("Unemployment Shock (pp)", -5.0, 15.0, 0.0, 0.5)
        shock_pi = st.slider("Inflation Shock (pp)", -5.0, 10.0, 0.0, 0.5)
        shock_horizon = st.slider("Horizon (quarters)", 4, 20, 8)
        decay = st.slider("Shock Decay Rate", 0.3, 1.0, 0.7, 0.05,
                          help="Fraction of shock remaining each period. "
                               "1.0 = permanent, 0.5 = halves each quarter.")

    with col2:
        st.subheader("Baseline")
        st.metric("Base Unemployment", f"{base_u:.2f}%")
        st.metric("Base Inflation", f"{base_pi:.2f}%")
        st.metric("Base PD", f"{result['predicted_pd']:.2%}")
        st.metric("Base Rating", result["rating"])

    credit = CreditModel()
    shock_path = build_shock_path(base_u, base_pi, shock_u, shock_pi,
                                   shock_horizon, decay)
    shock_path = compute_pd_path(shock_path, credit)

    st.markdown("---")

    # Macro path chart
    fig_path = make_subplots(rows=1, cols=3,
                              subplot_titles=["Unemployment Path",
                                              "Inflation Path",
                                              "PD Path"])

    fig_path.add_trace(
        go.Scatter(x=shock_path["period"], y=shock_path["unemployment"],
                   mode="lines+markers", line=dict(color=COLORS["primary"], width=2),
                   name="Unemployment"),
        row=1, col=1
    )
    fig_path.add_hline(y=base_u, line_dash="dash", line_color="grey",
                        opacity=0.5, row=1, col=1, annotation_text="Base")

    fig_path.add_trace(
        go.Scatter(x=shock_path["period"], y=shock_path["inflation"],
                   mode="lines+markers", line=dict(color=COLORS["accent"], width=2),
                   name="Inflation"),
        row=1, col=2
    )
    fig_path.add_hline(y=base_pi, line_dash="dash", line_color="grey",
                        opacity=0.5, row=1, col=2, annotation_text="Base")

    # PD path with rating coloring
    for _, row_data in shock_path.iterrows():
        color = RATING_COLOR.get(row_data["rating"], COLORS["grey"])
        fig_path.add_trace(
            go.Scatter(
                x=[row_data["period"]],
                y=[row_data["predicted_pd"] * 100],
                mode="markers",
                marker=dict(color=color, size=10),
                showlegend=False,
                hovertemplate=(
                    f"Period {int(row_data['period'])}<br>"
                    f"PD: {row_data['predicted_pd']:.2%}<br>"
                    f"Rating: {row_data['rating']}<extra></extra>"
                ),
            ),
            row=1, col=3
        )

    fig_path.add_trace(
        go.Scatter(x=shock_path["period"],
                   y=shock_path["predicted_pd"] * 100,
                   mode="lines", line=dict(color=COLORS["warning"], width=2),
                   name="PD"),
        row=1, col=3
    )

    fig_path.update_layout(height=380, showlegend=False,
                            yaxis3_ticksuffix="%",
                            yaxis3_title="PD (%)")
    st.plotly_chart(fig_path, use_container_width=True)

    # Table
    st.subheader("Shock Path Table")
    display = shock_path.copy()
    display["unemployment"] = display["unemployment"].map("{:.2f}%".format)
    display["inflation"] = display["inflation"].map("{:.2f}%".format)
    display["predicted_pd"] = display["predicted_pd"].map("{:.3%}".format)
    display.rename(columns={"period": "Quarter"}, inplace=True)
    st.dataframe(display, use_container_width=True, hide_index=True)

    # Sensitivity heatmap
    st.markdown("---")
    st.subheader("PD Sensitivity Heatmap")
    u_range = np.arange(1, 18, 0.5)
    pi_range = np.arange(-2, 12, 0.5)
    pd_grid = np.array([
        [credit.predict_pd(float(u), float(pi)) * 100 for u in u_range]
        for pi in pi_range
    ])
    fig_heat = go.Figure(go.Heatmap(
        z=pd_grid, x=u_range, y=pi_range,
        colorscale="RdYlGn_r",
        colorbar=dict(title="PD (%)"),
        hovertemplate="U: %{x}%<br>π: %{y}%<br>PD: %{z:.2f}%<extra></extra>",
        zmin=0, zmax=20,
    ))
    fig_heat.add_trace(go.Scatter(
        x=[base_u], y=[base_pi],
        mode="markers",
        marker=dict(size=14, color="white", symbol="star",
                    line=dict(color="black", width=2)),
        name="Base Forecast",
    ))
    if shock_u != 0 or shock_pi != 0:
        fig_heat.add_trace(go.Scatter(
            x=[base_u + shock_u], y=[base_pi + shock_pi],
            mode="markers",
            marker=dict(size=14, color=COLORS["warning"], symbol="x",
                        line=dict(color="black", width=1)),
            name="Shock Impact",
        ))
    fig_heat.update_layout(
        height=420,
        xaxis_title="Unemployment (%)",
        yaxis_title="Inflation (%)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "🌍 Scenario Comparison":
    st.title("🌍 Scenario Comparison")
    st.caption("Compare 2–5 macro worlds side by side: PD curves, IRFs, ratings.")

    col_ctrl, col_run = st.columns([3, 1])
    with col_ctrl:
        selected_scenarios = st.multiselect(
            "Select Scenarios to Compare",
            options=list(NAMED_SCENARIOS.keys()),
            default=["Baseline", "Adverse", "COVID Shock"],
        )
    with col_run:
        st.write("")
        st.write("")
        compare_btn = st.button("▶ Compare", use_container_width=True)

    if compare_btn and selected_scenarios:
        if len(selected_scenarios) < 2:
            st.warning("Select at least 2 scenarios.")
        else:
            with st.spinner("Running scenario comparison..."):
                try:
                    comparison = run_scenario_comparison(
                        scenarios=selected_scenarios,
                        model_type=model_type,
                        var_lags=var_lags,
                    )
                    st.session_state.comparison_results = comparison
                except Exception as e:
                    st.error(f"Comparison error: {e}")

    comparison = st.session_state.comparison_results

    if not comparison:
        st.info("Select scenarios and click ▶ Compare.")
        st.stop()

    scenario_names = list(comparison.keys())
    palette = px.colors.qualitative.Set2

    # ── KPI comparison table ─────────────────────────────────────────────────
    st.subheader("Summary")
    rows = []
    for name, res in comparison.items():
        rows.append({
            "Scenario": name,
            "Unemployment Shock": f"{NAMED_SCENARIOS[name]['shock_u']:+.1f}pp",
            "Inflation Shock": f"{NAMED_SCENARIOS[name]['shock_pi']:+.1f}pp",
            "Forecast U": f"{res['forecast']['unemployment']:.2f}%",
            "Forecast π": f"{res['forecast']['inflation']:.2f}%",
            "PD": f"{res['predicted_pd']:.2%}",
            "Rating": res["rating"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── PD comparison bar ────────────────────────────────────────────────────
    st.subheader("Probability of Default by Scenario")
    fig_pd = go.Figure()
    for i, (name, res) in enumerate(comparison.items()):
        r = res["rating"]
        fig_pd.add_trace(go.Bar(
            x=[name],
            y=[res["predicted_pd"] * 100],
            marker_color=RATING_COLOR.get(r, palette[i % len(palette)]),
            text=f"{res['predicted_pd']:.2%}<br>{r}",
            textposition="outside",
            name=name,
        ))
    fig_pd.update_layout(height=380, yaxis_title="PD (%)",
                          showlegend=False, yaxis_ticksuffix="%")
    st.plotly_chart(fig_pd, use_container_width=True)

    # ── Shock path PD curves ─────────────────────────────────────────────────
    st.subheader("PD Trajectory Over Shock Horizon")
    credit = CreditModel()
    fig_curves = go.Figure()

    for i, (name, res) in enumerate(comparison.items()):
        params = NAMED_SCENARIOS[name]
        base_u = res["forecast"]["unemployment"]
        base_pi = res["forecast"]["inflation"]
        path = build_shock_path(base_u, base_pi,
                                params["shock_u"], params["shock_pi"],
                                horizon=8, decay=0.7)
        path = compute_pd_path(path, credit)

        fig_curves.add_trace(go.Scatter(
            x=path["period"],
            y=path["predicted_pd"] * 100,
            mode="lines+markers",
            name=name,
            line=dict(color=palette[i % len(palette)], width=2.5),
            hovertemplate=f"{name}<br>Quarter %{{x}}<br>PD: %{{y:.2f}}%<extra></extra>",
        ))

    fig_curves.update_layout(
        height=380,
        yaxis_title="PD (%)",
        xaxis_title="Quarter",
        yaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    # ── IRF comparison ───────────────────────────────────────────────────────
    st.subheader("IRF Comparison: Unemployment Response to Unemployment Shock")
    fig_irf_comp = go.Figure()

    for i, (name, res) in enumerate(comparison.items()):
        irfs = res.get("irf", {}).get("irfs", [])
        if not irfs:
            continue
        vals = [float(irfs[h][0, 0]) for h in range(len(irfs))]
        fig_irf_comp.add_trace(go.Scatter(
            x=list(range(len(vals))),
            y=vals,
            mode="lines+markers",
            name=name,
            line=dict(color=palette[i % len(palette)], width=2),
        ))

    fig_irf_comp.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
    fig_irf_comp.update_layout(
        height=350,
        xaxis_title="Horizon (quarters)",
        yaxis_title="Response",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_irf_comp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: RUN HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif tab_selected == "🕘 Run History":
    st.title("🕘 Run History & Replay")

    history = st.session_state.run_history

    if not history:
        st.info("No runs yet this session. Run the pipeline to start logging.")
        st.stop()

    history_df = pd.DataFrame(history)
    history_df["pd"] = history_df["pd"].map("{:.2%}".format)
    history_df["u_forecast"] = history_df["u_forecast"].map("{:.2f}%".format)
    history_df["pi_forecast"] = history_df["pi_forecast"].map("{:.2f}%".format)
    history_df.columns = ["Run ID", "Model", "Lags", "PD", "Rating",
                           "U Forecast", "π Forecast"]

    st.subheader("Session Run Log")
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    # Replay
    st.markdown("---")
    st.subheader("Replay a Run")
    run_ids = [r["id"] for r in history]
    selected_id = st.selectbox("Select Run ID", run_ids)

    if st.button("🔁 Replay Selected Run"):
        selected_run = next(r for r in history if r["id"] == selected_id)
        with st.spinner("Replaying..."):
            try:
                replayed = run_pipeline(
                    model_type=selected_run["model"],
                    var_lags=selected_run["lags"],
                    run_stress=True,
                )
                st.success(f"Replayed run {selected_id}")
                fc = replayed["forecast"]
                r1, r2, r3 = st.columns(3)
                r1.metric("PD", f"{replayed['predicted_pd']:.2%}")
                r2.metric("Rating", replayed["rating"])
                r3.metric("Model", replayed["model_type"])
            except Exception as e:
                st.error(f"Replay error: {e}")

    # Compare two runs
    if len(history) >= 2:
        st.markdown("---")
        st.subheader("Compare Two Runs")
        col_a, col_b = st.columns(2)
        with col_a:
            run_a_id = st.selectbox("Run A", run_ids, key="run_a")
        with col_b:
            run_b_id = st.selectbox("Run B", run_ids,
                                     index=min(1, len(run_ids) - 1), key="run_b")

        if st.button("📊 Compare"):
            run_a = next(r for r in history if r["id"] == run_a_id)
            run_b = next(r for r in history if r["id"] == run_b_id)

            comp_data = {
                "Metric": ["Model", "Lags", "U Forecast", "π Forecast", "PD", "Rating"],
                f"Run {run_a_id}": [
                    run_a["model"], run_a["lags"],
                    run_a["u_forecast"], run_a["pi_forecast"],
                    run_a["pd"], run_a["rating"],
                ],
                f"Run {run_b_id}": [
                    run_b["model"], run_b["lags"],
                    run_b["u_forecast"], run_b["pi_forecast"],
                    run_b["pd"], run_b["rating"],
                ],
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)