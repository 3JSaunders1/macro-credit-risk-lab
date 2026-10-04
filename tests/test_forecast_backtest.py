"""Forecast backtest: no look-ahead, and a correct Diebold-Mariano test."""
import numpy as np
import pandas as pd
from analysis.forecast_backtest import diebold_mariano, rolling_forecasts


def test_dm_is_zero_for_identical_errors():
    e = np.random.default_rng(0).normal(size=100)
    stat, p = diebold_mariano(e, e.copy(), h=1)
    assert abs(stat) < 1e-6 and p > 0.99


def test_dm_negative_when_first_model_is_better():
    rng = np.random.default_rng(1)
    stat, p = diebold_mariano(rng.normal(0, 0.5, 300), rng.normal(0, 2.0, 300), h=1)
    assert stat < 0 and p < 0.01


def test_rolling_forecasts_never_use_future_data():
    rng = np.random.default_rng(2)
    idx = pd.date_range("1990-01-01", periods=60, freq="QS")
    df = pd.DataFrame({"unemployment": 5 + rng.normal(0, 0.3, 60).cumsum() * 0.1,
                       "inflation": 2.5 + rng.normal(0, 0.3, 60)}, index=idx)
    fc = rolling_forecasts(df, horizons=[1, 4])
    assert (fc["target"] > fc["origin"]).all()
    offsets = (fc["target"].dt.to_period("Q") - fc["origin"].dt.to_period("Q")).apply(lambda x: x.n)
    assert (offsets == fc["h"]).all()
