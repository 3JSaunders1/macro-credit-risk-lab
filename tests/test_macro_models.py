"""Macro models: BVAR prior, intervals, identification, LP accuracy, and FEVD."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR
from models.bvar import BVARModel
from models.svar_cholesky import SVARCholesky
from models.sign_restriction_svar import SVARSignRestrictions
from models.local_projections import LocalProjections
from models.var_model import VARModel
from models.var_utils import fevd_from_irfs

A = np.array([[0.6, 0.1], [-0.2, 0.5]])
SIGMA = np.array([[1.0, -0.3], [-0.3, 1.0]])


@pytest.fixture(scope="module")
def var_data():
    rng = np.random.default_rng(0)
    T, y = 600, np.zeros((600, 2))
    L = np.linalg.cholesky(SIGMA)
    for t in range(1, T):
        y[t] = A @ y[t - 1] + L @ rng.normal(size=2)
    return pd.DataFrame(y[100:] + [5.0, 2.5], columns=["unemployment", "inflation"])


def test_bvar_tight_prior_shrinks_to_prior(var_data):
    m = BVARModel(lags=2, lambda_=1e-4).fit(var_data)
    A1, A2 = m.lag_matrices()
    assert abs(A1[0, 1]) < 0.01 and abs(A1[1, 0]) < 0.01      # cross terms -> 0
    assert np.all(np.abs(A2) < 0.01)                          # longer lags -> 0
    assert m.diagnostics()["var_stability"]


def test_bvar_loose_prior_matches_ols(var_data):
    bvar = BVARModel(lags=1, lambda_=1e4).fit(var_data)
    ols = VAR(var_data).fit(1)
    assert np.allclose(bvar.lag_matrices()[0], ols.coefs[0], atol=1e-3)


@pytest.mark.parametrize("model", [
    VARModel(lags=1), SVARCholesky(lags=1), BVARModel(lags=1),
    SVARSignRestrictions(lags=1, n_draws=300), LocalProjections(horizon=4, lags=1)])
def test_forecast_intervals_are_valid_and_widen(model, var_data):
    model.fit(var_data)
    point, lower, upper = model.forecast_interval(steps=4, alpha=0.10)
    assert np.all(lower < point) and np.all(point < upper)
    assert np.all((upper - lower)[3] > (upper - lower)[0])


def test_cholesky_impact_is_lower_triangular(var_data):
    impact = SVARCholesky(lags=1).fit(var_data).irf(periods=4).irfs[0]
    assert abs(impact[0, 1]) < 1e-10      # unemployment doesn't respond to inflation shock on impact


def test_sign_restrictions_reproducible_and_satisfied(var_data):
    a = SVARSignRestrictions(lags=1, n_draws=500, seed=7).fit(var_data)
    b = SVARSignRestrictions(lags=1, n_draws=500, seed=7).fit(var_data)
    assert np.allclose(np.array(a.irf().irfs), np.array(b.irf().irfs))
    assert a.acceptance_rate > 0
    med = np.array(a.irf().irfs)
    assert np.all(med[:3, 0, 0] > 0) and np.all(med[:3, 1, 0] < 0)


def test_local_projections_match_var_impulse_responses(var_data):
    lp = LocalProjections(horizon=3, lags=1).fit(var_data)
    chol = SVARCholesky(lags=1).fit(var_data).irf(periods=3)
    for h in range(2):
        assert np.allclose(lp.irfs_[h], chol.irfs[h], atol=0.15)


def test_fevd_rows_sum_to_one():
    irfs = np.random.default_rng(1).normal(size=(5, 2, 2))
    assert np.allclose(fevd_from_irfs(irfs).sum(axis=2), 1.0)
