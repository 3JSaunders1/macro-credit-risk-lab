"""
models/credit_model.py
----------------------
Macro-driven credit loss model.

Maps macro variables to the consumer charge-off rate (a loss rate, roughly PD x LGD)
using a logistic form, estimated by fractional logit in models/credit_estimation.py.
Used for stress testing, rating assignment, and sensitivity analysis.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import settings

COEF_FILE = Path(__file__).resolve().parents[1] / "config" / "credit_coefficients.json"


def load_coefficients() -> tuple[float, float, float]:
    """Estimated coefficients if available; otherwise the original calibrated priors."""
    if COEF_FILE.exists():
        c = json.loads(COEF_FILE.read_text())
        return c["beta_0"], c["beta_u"], c["beta_pi"]
    return settings.BETA_0, settings.BETA_U, settings.BETA_PI


@dataclass
class StressScenario:
    name: str
    unemployment: float
    inflation: float


class CreditModel:
    def __init__(self, beta_0=None, beta_u=None, beta_pi=None):
        est_0, est_u, est_pi = load_coefficients()
        self.beta_0 = est_0 if beta_0 is None else beta_0
        self.beta_u = est_u if beta_u is None else beta_u
        self.beta_pi = est_pi if beta_pi is None else beta_pi

    # -------------------------
    # Core prediction
    # -------------------------
    def _logit(self, u, pi):
        return self.beta_0 + self.beta_u * u + self.beta_pi * pi

    def predict_pd(self, u, pi):
        z = self._logit(u, pi)
        return float(1 / (1 + np.exp(-z)))

    def predict_pd_batch(self, df):
        z = (
            self.beta_0
            + self.beta_u * df["unemployment"]
            + self.beta_pi * df["inflation"]
        )
        return 1 / (1 + np.exp(-z))

    # -------------------------
    # Estimation
    # -------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Estimate coefficients by fractional logit on the original scale.

        X: columns 'unemployment' and 'inflation'; y: rates between 0 and 1.
        Fitting on the original (unstandardized) scale avoids the intercept error
        in the earlier version, which standardized features without converting
        the intercept back.
        """
        import statsmodels.api as sm
        res = sm.GLM(y, sm.add_constant(X[["unemployment", "inflation"]]),
                     family=sm.families.Binomial()).fit()
        self.beta_0 = float(res.params["const"])
        self.beta_u = float(res.params["unemployment"])
        self.beta_pi = float(res.params["inflation"])
        return self

    # -------------------------
    # Rating
    # -------------------------
    def pd_to_rating(self, pd_val):
        if pd_val < 0.002: return "AAA"
        if pd_val < 0.005: return "AA"
        if pd_val < 0.01: return "A"
        if pd_val < 0.02: return "BBB"
        if pd_val < 0.05: return "BB"
        if pd_val < 0.1: return "B"
        if pd_val < 0.2: return "CCC"
        return "D"

    # -------------------------
    # Stress testing
    # -------------------------
    def stress_test(self, scenarios):
        rows = []
        for s in scenarios:
            pd_val = self.predict_pd(s.unemployment, s.inflation)
            rows.append({
                "scenario": s.name,
                "unemployment": s.unemployment,
                "inflation": s.inflation,
                "predicted_pd": pd_val,
                "rating": self.pd_to_rating(pd_val),
            })
        return pd.DataFrame(rows)

    # -------------------------
    # Sensitivity
    # -------------------------
    def sensitivity(self, u=4.0, pi=2.5, shock=1.0):
        base = self.predict_pd(u, pi)
        up = self.predict_pd(u + shock, pi)
        ip = self.predict_pd(u, pi + shock)
        return {
            "base_pd": base,
            "unemployment_shock_delta": up - base,
            "inflation_shock_delta": ip - base,
        }