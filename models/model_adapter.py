"""
models/model_adapter.py
-----------------------
Unifies all models into one output format: forecast, forecast intervals, IRFs, FEVD.
Every model implements forecast_interval(steps, alpha) -> (point, lower, upper).
"""


class ModelAdapter:

    def __init__(self, model, model_type: str, alpha: float = 0.1):
        self.model = model
        self.model_type = model_type
        self.alpha = alpha

    def forecast(self):
        point, lower, upper = self.model.forecast_interval(steps=1, alpha=self.alpha)
        return {
            "forecast": {"unemployment": float(point[0][0]), "inflation": float(point[0][1])},
            "forecast_ci": {
                "unemployment": {"lower": float(lower[0][0]), "upper": float(upper[0][0])},
                "inflation": {"lower": float(lower[0][1]), "upper": float(upper[0][1])},
            },
        }

    def irf(self):
        irf_obj = self.model.irf()
        return {"irfs": irf_obj.irfs, "fevd": irf_obj.fevd, "model_type": self.model_type}

    def diagnostics(self):
        if hasattr(self.model, "diagnostics"):
            return self.model.diagnostics()
        return {"var_lags_used": None, "var_stability": None}
