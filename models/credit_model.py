"""
models/credit_model.py
----------------------
Macro-driven Probability of Default (PD) model.

Maps macro variables → credit risk using a logistic function.
Used for stress testing, rating assignment, and sensitivity analysis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from config.settings import settings


@dataclass
class StressScenario:
    name: str
    unemployment: float
    inflation: float


class CreditModel:
    def __init__(
        self,
        beta_0=settings.BETA_0,
        beta_u=settings.BETA_U,
        beta_pi=settings.BETA_PI,
    ):
        self.beta_0 = beta_0
        self.beta_u = beta_u
        self.beta_pi = beta_pi

    # -------------------------
    # Core PD
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
    # Calibration
    # -------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        model = LogisticRegression()
        model.fit(Xs, y)

        self.beta_u = model.coef_[0][0] / scaler.scale_[0]
        self.beta_pi = model.coef_[0][1] / scaler.scale_[1]
        self.beta_0 = model.intercept_[0]

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
    # Sensitivity (FIXED CONTRACT)
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