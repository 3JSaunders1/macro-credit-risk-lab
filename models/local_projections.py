"""
models/local_projections.py
---------------------------
Local Projections (Jorda 2005): impulse responses estimated horizon by horizon.

For each horizon h and variable i:
    y_{i,t+h} = a_h + sum_j b_{h,j} * shock_{j,t} + sum_{l=1}^{p} G_{h,l} y_{t-l} + e_{t+h}

Shocks are Cholesky-orthogonalized VAR(p) residuals; lagged values of all variables
are included as controls. The IRF at horizon h is b_h. HAC standard errors.

Local projections do not produce forecasts, so forecast() carries the last value
forward (a random walk), with intervals from historical h-quarter changes.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from models.interfaces import IRFBundle


class LocalProjections:

    def __init__(self, horizon: int = 12, lags: int = 2):
        self.horizon = horizon
        self.lags = lags
        self.data = None
        self.irfs_ = None
        self.se_ = None

    def fit(self, data: pd.DataFrame) -> "LocalProjections":
        self.data = data
        arr = data.values.astype(float)
        T, n, p = arr.shape[0], arr.shape[1], self.lags

        # Cholesky-orthogonalized shocks from a VAR(p); row k corresponds to time p + k
        Y = arr[p:, :]
        X = np.hstack([np.ones((T - p, 1))] + [arr[p - l: T - l, :] for l in range(1, p + 1)])
        resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
        sigma = resid.T @ resid / max(T - p - X.shape[1], 1)
        try:
            P = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            P = np.eye(n)
        shocks = resid @ np.linalg.inv(P).T

        irfs, se = np.zeros((self.horizon, n, n)), np.zeros((self.horizon, n, n))
        for h in range(self.horizon):
            t_idx = np.arange(p, T - h)                      # time t of each observation
            if len(t_idx) < 2 * (1 + n + n * p):
                break
            controls = [arr[t_idx - l] for l in range(1, p + 1)]
            X_lp = np.column_stack([np.ones(len(t_idx)), shocks[t_idx - p]] + controls)
            for i in range(n):
                res = OLS(arr[t_idx + h, i], X_lp).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
                irfs[h, i, :] = res.params[1: 1 + n]
                se[h, i, :] = res.bse[1: 1 + n]

        self.irfs_, self.se_ = irfs, se
        return self

    def irf(self, horizon: int = None) -> IRFBundle:
        h = min(horizon or self.horizon, len(self.irfs_))
        return IRFBundle(irfs=[self.irfs_[i] for i in range(h)], fevd=None,
                         model_type="local_projections")

    def forecast(self, steps: int = 1) -> np.ndarray:
        return np.tile(self.data.values[-1:, :].astype(float), (steps, 1))

    def forecast_interval(self, steps: int = 1, alpha: float = 0.10):
        """Random-walk forecast with intervals from historical k-quarter changes."""
        arr = self.data.values.astype(float)
        point = self.forecast(steps)
        z = norm.ppf(1 - alpha / 2)
        sd = np.array([np.std(arr[k:] - arr[:-k], axis=0, ddof=1) for k in range(1, steps + 1)])
        return point, point - z * sd, point + z * sd

    def get_ci(self, alpha: float = 0.10):
        z = norm.ppf(1 - alpha / 2)
        return self.irfs_ - z * self.se_, self.irfs_ + z * self.se_

    def diagnostics(self) -> dict:
        return {"var_lags_used": self.lags, "var_stability": None, "lp_horizon": self.horizon}
