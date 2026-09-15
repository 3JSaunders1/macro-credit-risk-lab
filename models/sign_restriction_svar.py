import numpy as np
from statsmodels.tsa.api import VAR
from numpy.linalg import qr


class SVARSignRestrictions:

    def __init__(self, lags=2, horizon=10, n_draws=200):
        self.lags = lags
        self.horizon = horizon
        self.n_draws = n_draws

        self.results = None
        self.accepted = []

    def fit(self, data):
        self.results = VAR(data).fit(self.lags)
        self._estimate()
        return self

    def _rand_q(self, k):
        q, _ = qr(np.random.normal(size=(k, k)))
        return q

    def _estimate(self):

        base = self.results.irf(self.horizon).irfs
        k = self.results.neqs

        accepted = []

        for _ in range(self.n_draws):

            Q = self._rand_q(k)

            # structural transform
            candidate = np.einsum("hij,jk->hik", base, Q)

            # ----------------------------
            # STRONGER SIGN RESTRICTIONS
            # ----------------------------
            u_shock = candidate[:, 0, 0]   # unemployment response to u-shock
            pi_shock = candidate[:, 1, 0]  # inflation response to u-shock

            # enforce across multiple horizons (CRITICAL FIX)
            if (
                np.all(u_shock[:3] > 0) and
                np.all(pi_shock[:3] < 0)
            ):
                accepted.append(candidate)

        if len(accepted) == 0:
            accepted = [base]

        self.accepted = accepted

    def forecast(self, steps=1):
        return self.results.forecast(
            self.results.endog[-self.lags:], steps
        )

    def irf(self):
        from models.interfaces import IRFBundle

        avg = np.mean(np.array(self.accepted), axis=0)
        fevd = self.results.fevd(self.horizon).decomp
        irfs = [avg[i] for i in range(avg.shape[0])]

        return IRFBundle(
            irfs=irfs,
            fevd=fevd,
            model_type="svar_sign"
        )

    def diagnostics(self):
        return {
            "var_lags_used": self.lags,
            "var_stability": None,
        }