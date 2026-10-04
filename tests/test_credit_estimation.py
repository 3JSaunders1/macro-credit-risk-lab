# pylint: disable=no-name-in-module
"""Credit link: the fit recovers true coefficients, including the intercept."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from models.credit_model import CreditModel

def synthetic(beta_0=-5.0, beta_u=0.25, beta_pi=0.05, n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"unemployment": rng.uniform(3, 12, n),
                      "inflation": rng.uniform(-1, 8, n)})
    y = pd.Series(expit(beta_0 + beta_u * X["unemployment"] + beta_pi * X["inflation"]))
    return X, y


@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.PerfectSeparationWarning")
def test_fit_recovers_coefficients_including_intercept():
    X, y = synthetic()
    m = CreditModel(beta_0=0, beta_u=0, beta_pi=0).fit(X, y)
    assert m.beta_0 == pytest.approx(-5.0, abs=1e-3)   # the old version got this wrong
    assert m.beta_u == pytest.approx(0.25, abs=1e-3)
    assert m.beta_pi == pytest.approx(0.05, abs=1e-3)


@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.PerfectSeparationWarning")
def test_fitted_model_predicts_rates():
    X, y = synthetic()
    m = CreditModel(beta_0=0, beta_u=0, beta_pi=0).fit(X, y)
    p = m.predict_pd(8.0, 3.0)
    assert 0 < p < 1
    assert m.predict_pd(10.0, 3.0) > m.predict_pd(5.0, 3.0)


def test_explicit_coefficients_override_file():
    m = CreditModel(beta_0=-6.0, beta_u=0.18, beta_pi=0.10)
    assert (m.beta_0, m.beta_u, m.beta_pi) == (-6.0, 0.18, 0.10)