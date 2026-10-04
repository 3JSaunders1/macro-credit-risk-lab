"""
models/sign_restriction_svar.py
-------------------------------
Structural VAR identified by sign restrictions.

A demand-type unemployment shock is defined by its signs: it raises unemployment
and lowers inflation for the first `restrict_horizons` quarters. Random orthogonal
rotations of the Cholesky responses are drawn, and only rotations satisfying the
signs are kept. Results report the pointwise median with 16th-84th percentile bands.
"""
import numpy as np
from statsmodels.tsa.api import VAR
from models.interfaces import IRFBundle
from models.var_utils import fevd_from_irfs


class SVARSignRestrictions:

    def __init__(self, lags=2, horizon=10, n_draws=1000, seed=42, restrict_horizons=3):
        self.lags = lags
        self.horizon = horizon
        self.n_draws = n_draws
        self.seed = seed
        self.restrict_horizons = restrict_horizons
        self.results = None
        self.accepted = np.empty(0)
        self.acceptance_rate = 0.0

    def fit(self, data):
        self.results = VAR(data).fit(self.lags)
        self._estimate()
        return self

    def _random_rotation(self, rng, k):
        q, r = np.linalg.qr(rng.normal(size=(k, k)))
        return q @ np.diag(np.sign(np.diag(r)))          # uniform (Haar) rotation

    def _estimate(self):
        rng = np.random.default_rng(self.seed)
        base = self.results.irf(self.horizon).orth_irfs   # Cholesky responses, (H+1, n, n)
        k, r = self.results.neqs, self.restrict_horizons
        accepted = []
        for _ in range(self.n_draws):
            cand = np.einsum("hij,jk->hik", base, self._random_rotation(rng, k))
            if cand[0, 0, 0] < 0:                          # normalize the shock's sign
                cand[:, :, 0] *= -1
            if np.all(cand[:r, 0, 0] > 0) and np.all(cand[:r, 1, 0] < 0):
                accepted.append(cand)
        if not accepted:
            raise ValueError("No rotations satisfied the sign restrictions; "
                             "increase n_draws or relax the restrictions.")
        self.accepted = np.array(accepted)
        self.acceptance_rate = len(accepted) / self.n_draws

    def forecast(self, steps=1):
        return self.results.forecast(self.results.endog[-self.lags:], steps)

    def forecast_interval(self, steps=1, alpha=0.10):
        """Identification does not change reduced-form forecasts or their intervals."""
        return self.results.forecast_interval(
            self.results.endog[-self.lags:], steps, alpha=alpha)

    def irf_bands(self, lower_pct=16, upper_pct=84):
        return (np.percentile(self.accepted, lower_pct, axis=0),
                np.percentile(self.accepted, upper_pct, axis=0))

    def irf(self):
        median = np.median(self.accepted, axis=0)
        return IRFBundle(irfs=[median[i] for i in range(median.shape[0])],
                         fevd=fevd_from_irfs(median), model_type="svar_sign")

    def diagnostics(self):
        return {"var_lags_used": self.lags,
                "var_stability": bool(self.results.is_stable()),
                "sign_acceptance_rate": self.acceptance_rate}
