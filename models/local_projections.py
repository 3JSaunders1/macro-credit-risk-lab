"""
models/local_projections.py
---------------------------
Local Projections (Jordà 2005) for Impulse Response Functions.

Unlike VAR-based IRFs which compound model errors across horizons,
Local Projections estimate each horizon h directly by regressing
y_{t+h} on current shocks. This is:

  - More robust to misspecification
  - Easier to add controls
  - The preferred method in modern empirical macro (Ramey 2016)
  - Used by the Fed and ECB for robustness checks alongside VARs

Each horizon h is a separate OLS regression:
  y_{i,t+h} = α_h + Σ_l β_l * X_{t-l} + ε_{t+h}

The IRF at horizon h is the coefficient on the shock variable.

Why include this:
  - Shows you know the IRF literature beyond just VAR
  - Provides a robustness check for your VAR IRFs
  - LP-IRFs are asymptotically equivalent to VAR IRFs under correct spec
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from models.interfaces import IRFBundle


class LocalProjections:
    """
    LP-IRF: horizon-by-horizon OLS estimation of impulse responses.

    Parameters
    ----------
    horizon : int   Maximum IRF horizon
    lags    : int   Number of control lags included in each regression
    """

    def __init__(self, horizon: int = 12, lags: int = 2):
        self.horizon = horizon
        self.lags = lags
        self.data = None
        self.irfs_ = None
        self.se_ = None
        self._columns = None
        self._last_obs = None

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame) -> "LocalProjections":
        """
        Estimate LP-IRFs for all variables responding to all shocks.

        Shock identification: Cholesky ordering (same as reduced-form VAR).
        The shock to variable j at time t is the j-th Cholesky residual.
        """
        self.data = data
        arr = data.values.astype(float)
        self._columns = list(data.columns)
        T, n = arr.shape
        p = self.lags

        # Step 1: Get reduced-form residuals from a VAR(p)
        Y = arr[p:, :]
        X_list = [arr[p - l: T - l, :] for l in range(1, p + 1)]
        X = np.hstack([np.ones((T - p, 1))] + X_list)

        B_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        resid = Y - X @ B_ols  # (T-p, n)

        # Step 2: Cholesky on residual covariance
        denom = max(T - p - n * p - 1, 1)
        Sigma = (resid.T @ resid) / denom
        try:
            P = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            P = np.eye(n)

        # Orthogonalized shocks
        shocks = resid @ np.linalg.inv(P).T  # (T-p, n)

        # Step 3: LP regression for each horizon h and each variable i
        irfs = np.zeros((self.horizon, n, n))
        se = np.zeros((self.horizon, n, n))

        for h in range(self.horizon):
            for i in range(n):
                # Assign y_h and shocks_h based on horizon
                if h == 0:
                    y_h = arr[p:, i]
                    shocks_h = shocks
                else:
                    if p + h > T:
                        break
                    y_h = arr[p + h: T, i]
                    shocks_h = shocks[: len(y_h), :]

                # Skip if not enough observations
                if len(y_h) < n + 5:
                    continue

                X_lp = add_constant(shocks_h)

                try:
                    res = OLS(y_h[: len(X_lp)], X_lp).fit(
                        cov_type="HAC", cov_kwds={"maxlags": h + 1}
                    )
                    # Coefficients 1..n are shock responses (skip intercept)
                    irfs[h, i, :] = res.params[1: n + 1]
                    se[h, i, :] = res.bse[1: n + 1]
                except Exception:
                    pass

        self.irfs_ = irfs
        self.se_ = se
        self._last_obs = arr[-self.lags:]

        return self

    # ------------------------------------------------------------------
    def irf(self, horizon: int = None) -> IRFBundle:
        if self.irfs_ is None:
            raise ValueError("Must call fit() first.")

        h = horizon or self.horizon
        irfs_list = [self.irfs_[i] for i in range(min(h, len(self.irfs_)))]

        return IRFBundle(
            irfs=irfs_list,
            fevd=None,  # LP doesn't naturally produce FEVD
            model_type="local_projections"
        )

    # ------------------------------------------------------------------
    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        LP doesn't naturally forecast — uses naive last-observation carry.
        Included so LP satisfies the same interface as VAR/BVAR.
        """
        if self.data is None:
            raise ValueError("Must call fit() first.")
        arr = self.data.values.astype(float)
        last = arr[-1:, :]
        return np.tile(last, (steps, 1))

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "var_lags_used": self.lags,
            "var_stability": None,
            "lp_horizon": self.horizon,
        }

    # ------------------------------------------------------------------
    def get_ci(self, alpha: float = 0.10):
        """
        Confidence bands via normal approximation: IRF ± z * SE.

        Returns
        -------
        lower, upper : np.ndarray, both shape (horizon, n, n)
        """
        if self.irfs_ is None:
            raise ValueError("Must call fit() first.")

        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2)
        lower = self.irfs_ - z * self.se_
        upper = self.irfs_ + z * self.se_
        return lower, upper