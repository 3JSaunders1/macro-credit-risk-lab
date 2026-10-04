"""
api/server.py
-------------
FastAPI REST API for the Macro-Driven Credit Risk Lab.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.scenarios import SCENARIOS
from models.credit_model import CreditModel
from pipeline.scenario_engine import scenario_summary
from services.pipeline_service import run_fresh_pipeline
from utils.serialization import to_json_safe

app = FastAPI(title="Macro Credit Risk API", version="2.0.0")


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
        keys = ["experiment_id", "model_type", "forecast", "forecast_ci", "predicted_pd",
                "rating", "sensitivity", "stationarity", "var_lags_used", "var_stability"]
        return JSONResponse(content=to_json_safe({k: result.get(k) for k in keys}))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=PDResponse)
def score(inputs: MacroInputs):
    """Illustrative static mapping from macro levels to a PD and rating."""
    model = CreditModel()
    pd_val = model.predict_pd(inputs.unemployment, inputs.inflation)
    return PDResponse(unemployment=inputs.unemployment, inflation=inputs.inflation,
                      predicted_pd=pd_val, rating=model.pd_to_rating(pd_val))


@app.get("/scenarios")
def scenarios():
    """Scenario definitions (config/scenarios.py)."""
    return SCENARIOS


@app.get("/stress_test")
def stress_test():
    """Model-based stress test: peak macro values, peak loss rate, and 9-quarter cumulative loss."""
    return scenario_summary().round(4).to_dict(orient="records")
