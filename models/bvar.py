"""
models/bvar.py
--------------
Bayesian VAR with Minnesota Prior.

The Minnesota prior (Litterman 1986) shrinks VAR coefficients toward
a random walk for each variable, preventing overfitting in short macro
samples. This is standard in central bank forecasting (NY Fed, ECB).

The prior assumes:
  - Own lags: shrink toward 1 for lag 1, 0 for higher lags
  - Cross-variable lags: shrink toward 0
  - Tighter = more shrinkage toward the prior (lambda controls this)

Why this matters for the portfolio:
  - Minnesota prior BVAR is the workhorse model at most central banks
  - Mentioning it in an interview at a credit/risk firm signals you know
    the forecasting literature, not just sklearn
"""

import numpy as np
import pandas as pd
from scipy import linalg
from models.interfaces import IRFBundle


class BVARModel:
    """
    Bayesian VAR with Minnesota prior via dummy observations.

    Parameters
    ----------
    lags    : int   Number of lags
    lambda_ : float Prior tightness (smaller = tighter/more shrinkage)
    """

    def __init__(self, lags: int = 2, lambda_: float = 0.2):
        self.lags = lags
        self.lambda_ = lambda_
        self.fitted = None
        self.coeffs = None
        self.sigma = None
        self.data = None

    # ------------------------------------------------------------------
    # Minnesota dummy observations (Banbura et al. 2010 approach)
    # ------------------------------------------------------------------
    def _build_minnesota_dummies(self, data: np.ndarray):
        """
        Construct dummy observation matrices encoding the Minnesota prior.
        Returns (Y_d, X_d) dummy data appended to real data for OLS.
        """
        n = data.shape[1]  # number of variables
        p = self.lags
        lam = self.lambda_

        # Residual variance estimate (AR(1) for each variable)
        ar_sigmas = []
        for i in range(n):
            y = data[1:, i]
            x = data[:-1, i].reshape(-1, 1)
            b = np.linalg.lstsq(x, y, rcond=None)[0]
            resid = y - x @ b
            ar_sigmas.append(np.std(resid))
        s = np.array(ar_sigmas)

        # Dummy Y: (n*p + n) × n
        # Dummy X: (n*p + n) × (n*p + 1)
        rows_coeff = n * p
        rows_sigma = n
        total = rows_coeff + rows_sigma

        Y_d = np.zeros((total, n))
        X_d = np.zeros((total, n * p + 1))

        # Coefficient dummies (shrink toward own-lag-1 = 1, others = 0)
        for lag in range(1, p + 1):
            for var in range(n):
                row = (lag - 1) * n + var
                Y_d[row, var] = s[var] / (lam * lag)
                col = (lag - 1) * n + var + 1  # +1 for intercept
                X_d[row, col] = s[var] / (lam * lag)

        # Sigma dummies (prior on residual variance)
        for var in range(n):
            row = rows_coeff + var
            Y_d[row, var] = s[var]
            X_d[row, 0] = 0  # no intercept in sigma dummies

        return Y_d, X_d

    # ------------------------------------------------------------------
    # Build OLS design matrices from data
    # ------------------------------------------------------------------
    def _build_ols_matrices(self, data: np.ndarray):
        T, n = data.shape
        p = self.lags

        Y = data[p:, :]
        X = np.ones((T - p, n * p + 1))  # intercept + lags

        for lag in range(1, p + 1):
            X[:, 1 + (lag - 1) * n: 1 + lag * n] = data[p - lag: T - lag, :]

        return Y, X

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> "BVARModel":
        self.data = data
        arr = data.values.astype(float)

        Y, X = self._build_ols_matrices(arr)
        Y_d, X_d = self._build_minnesota_dummies(arr)

        # Augment with dummies
        Y_aug = np.vstack([Y, Y_d])
        X_aug = np.vstack([X, X_d])

        # Posterior OLS: B = (X'X)^{-1} X'Y
        XtX = X_aug.T @ X_aug
        XtY = X_aug.T @ Y_aug

        self.coeffs = np.linalg.solve(XtX, XtY)  # shape: (n*p+1, n)

        resid = Y_aug - X_aug @ self.coeffs
        self.sigma = (resid.T @ resid) / (Y_aug.shape[0] - X_aug.shape[1])

        self._last_obs = arr[-self.lags:]
        self._n = arr.shape[1]
        self._columns = list(data.columns)

        return self

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------
    def forecast(self, steps: int = 1) -> np.ndarray:
        if self.coeffs is None:
            raise ValueError("Model must be fit before forecasting.")

        n, p = self._n, self.lags
        B = self.coeffs  # (n*p+1, n)

        history = list(self._last_obs)
        forecasts = []

        for _ in range(steps):
            x = np.ones(n * p + 1)
            for lag in range(p):
                x[1 + lag * n: 1 + (lag + 1) * n] = history[-(lag + 1)]
            y_hat = x @ B
            forecasts.append(y_hat)
            history.append(y_hat)

        return np.array(forecasts)  # (steps, n)

    # ------------------------------------------------------------------
    # IRF via Cholesky (reduced-form residuals)
    # ------------------------------------------------------------------
    def irf(self, horizon: int = 10) -> IRFBundle:
        if self.coeffs is None:
            raise ValueError("Must fit before computing IRF.")

        n, p = self._n, self.lags
        B = self.coeffs[1:, :].T  # (n, n*p) — drop intercept, transpose

        # Companion matrix
        k = n * p
        companion = np.zeros((k, k))
        companion[:n, :] = B
        companion[n:, :-n] = np.eye(k - n)

        # Cholesky structural identification
        try:
            P = np.linalg.cholesky(self.sigma)
        except np.linalg.LinAlgError:
            P = np.eye(n)

        irfs = []
        Ak = np.eye(k)
        for _ in range(horizon):
            response = Ak[:n, :n] @ P
            irfs.append(response)
            Ak = Ak @ companion

        # FEVD
        mse = np.zeros((n, n))
        fevd = np.zeros((horizon, n, n))
        for h in range(horizon):
            shock_contrib = np.array([
                irfs[h][:, j] ** 2 for j in range(n)
            ]).T  # (n, n)
            mse += shock_contrib
            fevd[h] = mse / (mse.sum(axis=1, keepdims=True) + 1e-10)

        return IRFBundle(
            irfs=[irfs[i] for i in range(horizon)],
            fevd=fevd,
            model_type="bvar_minnesota"
        )

    def diagnostics(self) -> dict:
        return {
            "var_lags_used": self.lags,
            "var_stability": None,  # TODO: check companion eigenvalues
            "bvar_lambda": self.lambda_,
        }