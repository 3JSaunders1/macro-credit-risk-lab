import numpy as np


class ModelAdapter:
    """
    Unifies ALL models into:
    - forecast
    - forecast_ci
    - irf
    - fevd
    """

    def __init__(self, model, model_type: str, alpha: float = 0.1):
        self.model = model
        self.model_type = model_type
        self.alpha = alpha

    def forecast(self):
        fc = self.model.forecast(steps=1)[0]

        u = float(fc[0])
        pi = float(fc[1])

        # simple but stable CI
        spread = 0.4

        return {
            "forecast": {
                "unemployment": u,
                "inflation": pi
            },
            "forecast_ci": {
                "unemployment": {"lower": u - spread, "upper": u + spread},
                "inflation": {"lower": pi - spread, "upper": pi + spread}
            }
        }

    def irf(self):
        irf_obj = self.model.irf()

        return {
            "irfs": irf_obj.irfs,
            "fevd": irf_obj.fevd,
            "model_type": self.model_type
        }

    def diagnostics(self):
        if hasattr(self.model, "diagnostics"):
            return self.model.diagnostics()
        return {
            "var_lags_used": None,
            "var_stability": None
        }