"""
models/svar_cholesky.py
-----------------------
Structural VAR identified by Cholesky ordering (unemployment first, then inflation):
inflation can respond to an unemployment shock within the quarter, but not vice versa.
"""
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
        return self.results.forecast(self.results.endog[-self.lags:], steps)

    def forecast_interval(self, steps=1, alpha=0.10):
        """Identification does not change reduced-form forecasts or their intervals."""
        return self.results.forecast_interval(
            self.results.endog[-self.lags:], steps, alpha=alpha)

    def irf(self, periods=10):
        irf = self.results.irf(periods)
        irfs = [irf.orth_irfs[i] for i in range(irf.orth_irfs.shape[0])]   # orthogonalized
        fevd = np.transpose(self.results.fevd(periods).decomp, (1, 0, 2))   # (horizon, variable, shock)
        return IRFBundle(irfs=irfs, fevd=fevd, model_type="svar_cholesky")

    def diagnostics(self):
        return {"var_lags_used": self.lags,
                "var_stability": bool(self.results.is_stable())}
