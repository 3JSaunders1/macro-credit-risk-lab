import numpy as np
from statsmodels.tsa.api import VAR
from models.interfaces import IRFBundle


class SVARCholesky:

    def __init__(self, lags=2):
        self.lags = lags
        self.results = None

    def fit(self, data):
        self.results = VAR(data).fit(self.lags)
        return self

    def forecast(self, steps=1):
        return self.results.forecast(
            self.results.endog[-self.lags:], steps
        )

    def irf(self, periods=10):

        # FIX: no orth argument exists
        irf = self.results.irf(periods)

        fevd = self.results.fevd(periods).decomp

        irfs = [irf.irfs[i] for i in range(irf.irfs.shape[0])]

        return IRFBundle(
            irfs=irfs,
            fevd=fevd,
            model_type="svar_cholesky"
        )
    
    def diagnostics(self):
        return {
            "var_lags_used": self.lags,
            "var_stability": None,
        }