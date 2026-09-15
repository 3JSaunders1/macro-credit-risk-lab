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
        return self.fitted.forecast(
            self.fitted.endog[-self.lags:], steps
        )

    def irf(self, horizon=10):
        irf_obj = self.fitted.irf(horizon)
        fevd_obj = self.fitted.fevd(horizon)

        irfs = [
            irf_obj.irfs[i]
            for i in range(irf_obj.irfs.shape[0])
        ]

        return IRFBundle(
            irfs=irfs,
            fevd=fevd_obj.decomp,
            model_type="var"
        )

    def diagnostics(self):
        return {
            "var_lags_used": self.lags,
            "var_stability": bool(self.fitted.is_stable())
        }