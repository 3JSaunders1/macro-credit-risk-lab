"""
pipeline/run_pipeline.py
------------------------
Core pipeline: data → model → forecast → credit risk → stress test.

Supports:
  - VAR, SVAR (Cholesky, Sign), BVAR, Local Projections
  - IRF + FEVD for all model types
  - Shock path engine (time series stress trajectories)
  - Scenario comparison (run multiple worlds)
  - Stationarity diagnostics
  - Experiment logging
"""

import uuid
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from config.settings import settings
from models.credit_model import CreditModel, StressScenario
from models.model_adapter import ModelAdapter
from models.var_model import VARModel
from models.svar_cholesky import SVARCholesky
from models.sign_restriction_svar import SVARSignRestrictions
from models.bvar import BVARModel
from models.local_projections import LocalProjections


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(path: str = settings.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[["unemployment", "inflation"]].dropna().astype(float)
    return df


def check_stationarity(data: pd.DataFrame, sig: float = 0.05) -> dict:
    results = {}
    for col in data.columns:
        stat, p, *_ = adfuller(data[col].dropna())
        results[col] = {
            "adf_stat": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "stationary": bool(p < sig),
        }
    return results


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(model_type: str, var_lags: int, data: pd.DataFrame):
    if model_type == "var":
        return VARModel(lags=var_lags).fit(data)
    elif model_type == "svar_cholesky":
        return SVARCholesky(lags=var_lags).fit(data)
    elif model_type == "svar_sign":
        return SVARSignRestrictions(lags=var_lags).fit(data)
    elif model_type == "bvar":
        return BVARModel(lags=var_lags, lambda_=settings.BVAR_LAMBDA).fit(data)
    elif model_type == "local_projections":
        return LocalProjections(
            horizon=settings.LP_HORIZON, lags=var_lags
        ).fit(data)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def normalize_forecast(fc: np.ndarray) -> tuple[float, float]:
    fc = np.asarray(fc)
    if fc.ndim == 1:
        u, pi = fc[0], fc[1]
    else:
        u, pi = fc[0, 0], fc[0, 1]
    return float(np.clip(u, -10, 30)), float(np.clip(pi, -10, 30))


# ── Shock path engine ─────────────────────────────────────────────────────────

def build_shock_path(
    base_u: float,
    base_pi: float,
    shock_u: float = 0.0,
    shock_pi: float = 0.0,
    horizon: int = 8,
    decay: float = 0.7,
) -> pd.DataFrame:
    """
    Build a time-series shock trajectory with exponential decay.

    Returns a DataFrame of shape (horizon, 3):
        [period, unemployment, inflation]

    decay: fraction of shock remaining each period (0.7 = 30% dissipates/period)
    """
    periods = []
    for h in range(horizon):
        factor = decay ** h
        periods.append({
            "period": h,
            "unemployment": base_u + shock_u * factor,
            "inflation": base_pi + shock_pi * factor,
        })
    return pd.DataFrame(periods)


def compute_pd_path(shock_path: pd.DataFrame, credit_model: CreditModel) -> pd.DataFrame:
    """Apply credit model along a shock path to get PD over time."""
    path = shock_path.copy()
    path["predicted_pd"] = credit_model.predict_pd_batch(path)
    path["rating"] = path["predicted_pd"].apply(credit_model.pd_to_rating)
    return path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    data_path: str = settings.DATA_PATH,
    run_stress: bool = True,
    var_lags: int = 2,
    alpha: float = 0.10,
    shock: float = 1.0,
    model_type: str = "var",
    shock_u: float = 0.0,
    shock_pi: float = 0.0,
    shock_horizon: int = 8,
    shock_decay: float = 0.7,
) -> dict:
    """
    Full pipeline execution.

    Returns dict with keys:
        experiment_id, model_type, forecast, forecast_ci,
        irf, predicted_pd, rating, sensitivity, stress_test,
        stationarity, var_lags_used, var_stability,
        shock_path (if shocks specified)
    """
    df = load_data(data_path)
    stationarity = check_stationarity(df)

    model = build_model(model_type, var_lags, df)
    adapter = ModelAdapter(model, model_type, alpha)

    forecast_block = adapter.forecast()
    irf_block = adapter.irf()
    diagnostics = adapter.diagnostics()

    raw_fc = model.forecast(steps=1)
    u_hat, pi_hat = normalize_forecast(raw_fc)

    forecast_block["forecast"]["unemployment"] = u_hat
    forecast_block["forecast"]["inflation"] = pi_hat

    credit = CreditModel()
    pd_hat = credit.predict_pd(u_hat, pi_hat)
    rating = credit.pd_to_rating(pd_hat)
    sensitivity = credit.sensitivity(u=u_hat, pi=pi_hat, shock=shock)

    stress_df = None
    if run_stress:
        scenarios = [
            StressScenario("Baseline", 4.0, 2.5),
            StressScenario("Adverse", 7.0, 4.0),
            StressScenario("Severe", 12.0, 1.5),
            StressScenario("Stagflation", 8.0, 9.0),
            StressScenario("COVID", 15.0, 0.5),
        ]
        stress_df = credit.stress_test(scenarios)

    # Shock path
    shock_path_df = None
    if shock_u != 0.0 or shock_pi != 0.0:
        shock_path_df = build_shock_path(
            u_hat, pi_hat, shock_u, shock_pi, shock_horizon, shock_decay
        )
        shock_path_df = compute_pd_path(shock_path_df, credit)

    # IRF guard
    irfs = irf_block.get("irfs", [])
    if not irfs:
        irf_block["irfs"] = []
        irf_block["fevd"] = None

    return {
        "experiment_id": str(uuid.uuid4())[:8],
        "model_type": model_type,
        "forecast": forecast_block["forecast"],
        "forecast_ci": forecast_block["forecast_ci"],
        "irf": irf_block,
        "predicted_pd": float(pd_hat),
        "rating": rating,
        "sensitivity": sensitivity,
        "stress_test": stress_df,
        "stationarity": stationarity,
        "shock_path": shock_path_df,
        **diagnostics,
    }


# ── Scenario comparison ───────────────────────────────────────────────────────

NAMED_SCENARIOS = {
    "Baseline":        {"shock_u": 0.0,  "shock_pi": 0.0},
    "Adverse":         {"shock_u": 3.0,  "shock_pi": 1.5},
    "Severely Adverse":{"shock_u": 8.0,  "shock_pi": -1.0},
    "Stagflation":     {"shock_u": 4.0,  "shock_pi": 6.5},
    "COVID Shock":     {"shock_u": 11.0, "shock_pi": -2.0},
}


def run_scenario_comparison(
    scenarios: list[str],
    model_type: str = "var",
    var_lags: int = 2,
    shock_horizon: int = 8,
    shock_decay: float = 0.7,
    data_path: str = settings.DATA_PATH,
) -> dict:
    """
    Run pipeline for multiple named scenarios and return side-by-side results.

    Returns dict keyed by scenario name, each value is a full pipeline result.
    """
    results = {}
    for name in scenarios:
        if name not in NAMED_SCENARIOS:
            continue
        params = NAMED_SCENARIOS[name]
        result = run_pipeline(
            data_path=data_path,
            model_type=model_type,
            var_lags=var_lags,
            shock_horizon=shock_horizon,
            shock_decay=shock_decay,
            run_stress=False,
            **params,
        )
        result["scenario_name"] = name
        results[name] = result

    return results