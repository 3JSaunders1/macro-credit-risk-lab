"""
pipeline/run_pipeline.py
------------------------
Core pipeline: data -> forecasting model -> forecast with intervals -> IRFs/FEVD
-> illustrative static credit mapping -> model-based stress test.
"""
import uuid

import pandas as pd
from statsmodels.tsa.stattools import adfuller

from config.settings import settings
from models.bvar import BVARModel
from models.credit_model import CreditModel
from models.local_projections import LocalProjections
from models.model_adapter import ModelAdapter
from models.sign_restriction_svar import SVARSignRestrictions
from models.svar_cholesky import SVARCholesky
from models.var_model import VARModel
from pipeline.scenario_engine import scenario_summary


def load_data(path: str = settings.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df[["unemployment", "inflation"]].dropna().astype(float)


def check_stationarity(data: pd.DataFrame, sig: float = 0.05) -> dict:
    results = {}
    for col in data.columns:
        stat, p, *_ = adfuller(data[col].dropna())
        results[col] = {"adf_stat": round(float(stat), 4), "p_value": round(float(p), 4),
                        "stationary": bool(p < sig)}
    return results


def build_model(model_type: str, var_lags: int, data: pd.DataFrame):
    if model_type == "var":
        return VARModel(lags=var_lags).fit(data)
    if model_type == "svar_cholesky":
        return SVARCholesky(lags=var_lags).fit(data)
    if model_type == "svar_sign":
        return SVARSignRestrictions(lags=var_lags).fit(data)
    if model_type == "bvar":
        return BVARModel(lags=var_lags, lambda_=settings.BVAR_LAMBDA).fit(data)
    if model_type == "local_projections":
        return LocalProjections(horizon=settings.LP_HORIZON, lags=var_lags).fit(data)
    raise ValueError(f"Unknown model_type: {model_type}")


def run_pipeline(data_path: str = settings.DATA_PATH, run_stress: bool = True,
                 var_lags: int = 2, alpha: float = 0.10, shock: float = 1.0,
                 model_type: str = "var") -> dict:
    df = load_data(data_path)
    model = build_model(model_type, var_lags, df)
    adapter = ModelAdapter(model, model_type, alpha)

    forecast_block = adapter.forecast()
    irf_block = adapter.irf()
    u_hat = forecast_block["forecast"]["unemployment"]
    pi_hat = forecast_block["forecast"]["inflation"]

    credit = CreditModel()                     # illustrative static mapping (calibrated priors)
    pd_hat = credit.predict_pd(u_hat, pi_hat)

    return {
        "experiment_id": str(uuid.uuid4())[:8],
        "model_type": model_type,
        "forecast": forecast_block["forecast"],
        "forecast_ci": forecast_block["forecast_ci"],
        "irf": irf_block,
        "predicted_pd": float(pd_hat),
        "rating": credit.pd_to_rating(pd_hat),
        "sensitivity": credit.sensitivity(u=u_hat, pi=pi_hat, shock=shock),
        "stress_test": scenario_summary() if run_stress else None,
        "stationarity": check_stationarity(df),
        **adapter.diagnostics(),
    }
