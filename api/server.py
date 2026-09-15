"""
api/server.py
-------------
FastAPI REST API exposing the Macro-Driven Credit Scoring System.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.pipeline_service import run_fresh_pipeline
from models.credit_model import CreditModel, StressScenario
from utils.serialization import to_json_safe

app = FastAPI(title="Macro Credit Risk API", version="1.0.0")


class MacroInputs(BaseModel):
    unemployment: float = Field(..., ge=0, le=30)
    inflation: float = Field(..., ge=-5, le=30)


class PDResponse(BaseModel):
    unemployment: float
    inflation: float
    predicted_pd: float
    rating: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast_and_score")
def forecast_and_score():
    try:
        result = run_fresh_pipeline(run_stress=False)
        safe = {
            "experiment_id": result["experiment_id"],
            "model_type":    result["model_type"],
            "forecast":      result["forecast"],
            "forecast_ci":   result["forecast_ci"],
            "predicted_pd":  result["predicted_pd"],
            "rating":        result["rating"],
            "sensitivity":   result["sensitivity"],
            "stationarity":  result["stationarity"],
            "var_lags_used": result.get("var_lags_used"),
            "var_stability": result.get("var_stability"),
        }
        return JSONResponse(content=to_json_safe(safe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=PDResponse)
def score(inputs: MacroInputs):
    model = CreditModel()
    pd_val = model.predict_pd(inputs.unemployment, inputs.inflation)
    return PDResponse(
        unemployment=inputs.unemployment,
        inflation=inputs.inflation,
        predicted_pd=pd_val,
        rating=model.pd_to_rating(pd_val),
    )


@app.get("/stress_test")
def stress_test():
    model = CreditModel()
    scenarios = [
        StressScenario("Baseline", 4, 2.5),
        StressScenario("Adverse", 7, 4),
        StressScenario("Severe", 12, 1.5),
        StressScenario("Stagflation", 8, 9),
        StressScenario("COVID", 15, 0.5),
    ]
    return model.stress_test(scenarios).to_dict(orient="records")