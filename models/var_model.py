"""
models/var_model.py
-------------------
Reduced-form VAR. IRFs are responses to one-unit reduced-form innovations;
see SVARCholesky for orthogonalized (structural) responses.
"""
import numpy as np
from statsmodels.tsa.api import VAR
from models.interfaces import IRFBundle


class VARModel:

    def __init__(self, lags=2, max_lags=8):
        self.lags = lags
        self.max_lags = max_lags
        self.fitted = None

    def fit(self, data):
        model = VAR(data)
        if self.lags is None:
            self.lags = model.select_order(self.max_lags).aic
        self.fitted = model.fit(self.lags)
        return self

    def forecast(self, steps=1):
        return self.fitted.forecast(self.fitted.endog[-self.lags:], steps)

    def forecast_interval(self, steps=1, alpha=0.10):
        point, lower, upper = self.fitted.forecast_interval(
            self.fitted.endog[-self.lags:], steps, alpha=alpha)
        return point, lower, upper

    def irf(self, horizon=10):
        irf_obj = self.fitted.irf(horizon)
        irfs = [irf_obj.irfs[i] for i in range(irf_obj.irfs.shape[0])]
        return IRFBundle(irfs=irfs, fevd=self.fitted.fevd(horizon).decomp, model_type="var")

    def diagnostics(self):
        return {"var_lags_used": self.lags,
                "var_stability": bool(self.fitted.is_stable())}
