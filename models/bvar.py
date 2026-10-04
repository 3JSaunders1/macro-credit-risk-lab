"""
models/bvar.py
--------------
Bayesian VAR with a Minnesota prior, implemented with dummy observations
(Banbura, Giannone and Reichlin 2010).

Prior: each variable follows its own AR(1) (coefficient delta_i on its own first
lag, estimated and capped at 1), every other coefficient is centered at zero,
and shrinkage tightens at longer lags (prior scale ~ lambda / lag). Smaller
lambda means more shrinkage toward the prior.
"""
import numpy as np
import pandas as pd
from models.interfaces import IRFBundle
from models.var_utils import (fevd_from_irfs, forecast_mse, is_stable,
                              ma_coefficients, normal_interval)


class BVARModel:

    def __init__(self, lags: int = 2, lambda_: float = 0.2):
        self.lags = lags
        self.lambda_ = lambda_
        self.coeffs = None
        self.sigma = None

    @staticmethod
    def _ar1_stats(arr: np.ndarray):
        """AR(1) coefficient and residual std for each variable (with intercept)."""
        deltas, sigmas = [], []
        for i in range(arr.shape[1]):
            y, x = arr[1:, i], np.column_stack([np.ones(len(arr) - 1), arr[:-1, i]])
            b = np.linalg.lstsq(x, y, rcond=None)[0]
            deltas.append(float(np.clip(b[1], 0, 1)))
            sigmas.append(float(np.std(y - x @ b, ddof=2)))
        return np.array(deltas), np.array(sigmas)

    def _build_minnesota_dummies(self, arr: np.ndarray):
        n, p, lam = arr.shape[1], self.lags, self.lambda_
        delta, s = self._ar1_stats(arr)
        Y_d = np.zeros((n * p + n, n))
        X_d = np.zeros((n * p + n, n * p + 1))
        for lag in range(1, p + 1):
            for i in range(n):
                row = (lag - 1) * n + i
                if lag == 1:
                    Y_d[row, i] = delta[i] * s[i] / lam      # prior mean only on own first lag
                X_d[row, 1 + (lag - 1) * n + i] = s[i] * lag / lam
        for i in range(n):                                    # residual variance dummies
            Y_d[n * p + i, i] = s[i]
        return Y_d, X_d

    def _build_ols_matrices(self, arr: np.ndarray):
        T, n, p = arr.shape[0], arr.shape[1], self.lags
        Y = arr[p:, :]
        X = np.ones((T - p, n * p + 1))
        for lag in range(1, p + 1):
            X[:, 1 + (lag - 1) * n: 1 + lag * n] = arr[p - lag: T - lag, :]
        return Y, X

    def fit(self, data: pd.DataFrame) -> "BVARModel":
        arr = data.values.astype(float)
        Y, X = self._build_ols_matrices(arr)
        Y_d, X_d = self._build_minnesota_dummies(arr)
        Y_aug, X_aug = np.vstack([Y, Y_d]), np.vstack([X, X_d])
        self.coeffs = np.linalg.solve(X_aug.T @ X_aug, X_aug.T @ Y_aug)   # (n*p+1, n)

        resid = Y - X @ self.coeffs                                      # real data only
        self.sigma = resid.T @ resid / max(Y.shape[0] - X.shape[1], 1)
        self._last_obs, self._n = arr[-self.lags:], arr.shape[1]
        return self

    def lag_matrices(self) -> np.ndarray:
        """A_1..A_p with shape (p, n, n), where A_l[i, j] is variable i's loading on j at lag l."""
        n, p = self._n, self.lags
        return np.array([self.coeffs[1 + l * n: 1 + (l + 1) * n, :].T for l in range(p)])

    def forecast(self, steps: int = 1) -> np.ndarray:
        n, p = self._n, self.lags
        history, out = list(self._last_obs), []
        for _ in range(steps):
            x = np.ones(n * p + 1)
            for lag in range(p):
                x[1 + lag * n: 1 + (lag + 1) * n] = history[-(lag + 1)]
            y_hat = x @ self.coeffs
            out.append(y_hat)
            history.append(y_hat)
        return np.array(out)

    def forecast_interval(self, steps: int = 1, alpha: float = 0.10):
        point = self.forecast(steps)
        mse = forecast_mse(self.lag_matrices(), self.sigma, steps)
        lower, upper = normal_interval(point, np.diagonal(mse, axis1=1, axis2=2), alpha)
        return point, lower, upper

    def irf(self, horizon: int = 10) -> IRFBundle:
        try:
            P = np.linalg.cholesky(self.sigma)
        except np.linalg.LinAlgError:
            P = np.eye(self._n)
        irfs = ma_coefficients(self.lag_matrices(), horizon) @ P
        return IRFBundle(irfs=[irfs[h] for h in range(horizon)],
                         fevd=fevd_from_irfs(irfs), model_type="bvar_minnesota")

    def diagnostics(self) -> dict:
        return {"var_lags_used": self.lags,
                "var_stability": is_stable(self.lag_matrices()),
                "bvar_lambda": self.lambda_}
