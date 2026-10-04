"""Dynamic loss model: parameter recovery and simulation behavior."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from models.loss_model import LossModel


def synthetic(n=300, a=-0.5, rho=0.85, b=0.15, c=0.02, seed=0):
    rng = np.random.default_rng(seed)
    d4u = rng.normal(0, 1, n)
    infl = rng.normal(2.5, 1, n)
    z = np.empty(n)
    z[0] = logit(0.03)
    for t in range(1, n):
        z[t] = a + rho * z[t - 1] + b * d4u[t] + c * infl[t]
    df = pd.DataFrame({"loss_rate": expit(z), "d4_unemployment": d4u, "inflation": infl},
                      index=pd.date_range("1950-01-01", periods=n, freq="QS"))
    df["lag_logit_loss"] = logit(df["loss_rate"]).shift(1)
    for col in ["bankruptcy_2005q4", "bankruptcy_2006q1", "covid_policy"]:
        df[col] = 0
    return df.dropna()


@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.PerfectSeparationWarning")
def test_recovers_parameters_and_drops_all_zero_indicators():
    m = LossModel().fit(synthetic())
    assert m.params["lag_logit_loss"] == pytest.approx(0.85, abs=1e-3)
    assert m.params["d4_unemployment"] == pytest.approx(0.15, abs=1e-3)
    assert "covid_policy" not in m.used


@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.PerfectSeparationWarning")
def test_rising_unemployment_raises_simulated_losses():
    m = LossModel().fit(synthetic())
    idx = pd.date_range("2030-01-01", periods=8, freq="QS")
    calm = pd.DataFrame({"d4_unemployment": 0.0, "inflation": 2.5}, index=idx)
    stress = pd.DataFrame({"d4_unemployment": 3.0, "inflation": 2.5}, index=idx)
    assert m.simulate(0.03, stress).iloc[-1] > m.simulate(0.03, calm).iloc[-1]


@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.PerfectSeparationWarning")
def test_simulation_stays_in_bounds_and_converges():
    m = LossModel().fit(synthetic())
    idx = pd.date_range("2030-01-01", periods=80, freq="QS")
    calm = pd.DataFrame({"d4_unemployment": 0.0, "inflation": 2.5}, index=idx)
    sim = m.simulate(0.10, calm)
    assert ((sim > 0) & (sim < 1)).all()
    assert abs(sim.iloc[-1] - sim.iloc[-2]) < 1e-4    # settles at a long-run level
