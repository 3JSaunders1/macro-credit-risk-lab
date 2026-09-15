# Macro-Driven Credit Risk Lab

A production-structured Python project that chains macroeconomic forecasting models (VAR, SVAR, BVAR, Local Projections) with a logistic Probability of Default (PD) model. The pipeline forecasts unemployment and inflation, then uses those forecasts to estimate credit risk, run stress scenarios, and analyze shock dynamics -- all wrapped in a FastAPI backend and an interactive Streamlit dashboard.

---

## Architecture

data/macro_data.csv (SQLite-backed)
       |
       v
 [Forecasting Model] -- VAR / SVAR (Cholesky) / SVAR (Sign Restrictions) /
       |                BVAR (Minnesota Prior) / Local Projections
       |                fits on (unemployment, inflation) time series
       |                forecasts ahead w/ confidence intervals
       |                computes IRFs + FEVD
       v
 [Credit Model] -- logistic PD: PD = sigma(b0 + b1*u + b2*pi)
       |              maps PD to internal rating (AAA -> D)
       |              runs stress scenarios (Baseline, Adverse, etc.)
       |              computes dynamic shock paths with decay
       v
  FastAPI / CLI / Streamlit Dashboard

---

## Quick Start

### Option 1: Local Python

# 1. Create virtual environment and install dependencies
python setup_env.py

# 2. Activate the environment
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Run the pipeline
python main.py

# 4. Start the API server
uvicorn api.server:app --reload --port 8000

# 5. Launch the interactive dashboard
streamlit run dashboard/app.py

### Option 2: Makefile (recommended)

make setup        # create venv + install
make run          # run CLI pipeline
make api          # start API with hot-reload
make dashboard    # launch Streamlit dashboard
make run-all      # start API + dashboard together
make test         # run all tests
make calibrate-demo  # calibrate credit model on synthetic data

---

## API Endpoints

Once the server is running at http://localhost:8000:

GET  /health                -- Liveness check
GET  /forecast_and_score    -- Full pipeline: forecast -> PD score
POST /score                 -- Score a custom macro scenario
GET  /stress_test           -- Standard DFAST-style scenarios

Interactive docs: http://localhost:8000/docs

### Example: Score a custom scenario

curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"unemployment": 8.5, "inflation": 4.0}'

Response:
{
  "unemployment": 8.5,
  "inflation": 4.0,
  "predicted_pd": 0.074821,
  "rating": "BB"
}

---

## Interactive Dashboard

The Streamlit dashboard (dashboard/app.py) provides:

- Overview -- live KPIs, macro history, PD gauge, stationarity diagnostics, stress test table
- IRF / FEVD -- impulse response functions and forecast error variance decomposition across models
- Shock Engine -- define custom unemployment/inflation shocks and watch PD evolve over time with decay, plus a full sensitivity heatmap
- Scenario Comparison -- compare 2-5 named macro scenarios side by side (PD, IRFs, trajectories)
- Run History -- session log of every pipeline run, with replay and side-by-side run comparison

Model type, lag order, confidence level, and shock size are all configurable from the sidebar in real time.

---

## Forecasting Models (models/)

VAR                       | var_model.py              | Standard reduced-form VAR, AIC lag selection, stability check
SVAR (Cholesky)           | svar_cholesky.py          | Structural identification via Cholesky ordering
SVAR (Sign Restrictions)  | sign_restriction_svar.py  | Identification via sign-restricted structural shocks (multi-horizon enforcement)
BVAR (Minnesota Prior)    | bvar.py                   | Bayesian VAR with Minnesota shrinkage -- the workhorse model at most central banks
Local Projections         | local_projections.py      | Horizon-by-horizon OLS IRFs (Jorda 2005), robust to VAR misspecification, HAC standard errors

All models implement a shared interface (models/interfaces.py) -- fit(), forecast(), irf() -- via ModelAdapter, so any model can be swapped into the pipeline or dashboard without code changes.

### Credit Model (models/credit_model.py)

- Logistic regression: PD = sigma(b0 + b1*unemployment + b2*inflation)
- Default coefficients set to economically sensible priors
- fit() method: estimate from historical default data via sklearn
- stress_test(): runs Baseline / Adverse / Severely Adverse / Stagflation / COVID-like scenarios
- sensitivity(): marginal PD impact of 1pp macro shocks
- pd_to_rating(): maps PD to simplified Basel-style rating scale

---

## Stress Scenarios

Baseline           | Unemployment 4.0%  | Inflation 2.5%
Adverse            | Unemployment 7.0%  | Inflation 4.0%
Severely Adverse   | Unemployment 12.0% | Inflation 1.5%
Stagflation        | Unemployment 8.0%  | Inflation 9.0%
COVID-like Shock   | Unemployment 15.0% | Inflation 0.5%

Loosely inspired by Federal Reserve DFAST scenario design.

---

## Experiment Tracking

Every pipeline run is logged to a local SQLite database (models/experiment_store.py, runs/experiments.db), capturing parameters, metrics, and artifacts for reproducibility -- plus a lightweight ModelRegistry (models/model_registry.py) for versioning trained model bundles.

---

## Testing

pytest tests/ -v          # full test suite
pytest tests/ -x          # stop on first failure
pytest tests/ --cov=.     # with coverage

Tests cover: credit model math, all five forecasting models (fit/forecast/IRF), full pipeline integration, shock paths, scenario comparison, and every API endpoint.

---

## Project Structure

macro-credit-risk-lab/
├── analysis/
│   └── compare_runs.py       # Compare IRFs, forecasts, and FEVD across runs
├── api/
│   └── server.py             # FastAPI app
├── config/
│   └── settings.py           # Centralized model/pipeline configuration
├── dashboard/
│   └── app.py                # Streamlit interactive dashboard
├── data/
│   └── macro_data.csv        # Quarterly macro data
├── models/
│   ├── base_model.py         # Abstract model interface
│   ├── bvar.py                # Bayesian VAR (Minnesota prior)
│   ├── credit_model.py        # Logistic PD model + stress testing
│   ├── experiment_store.py    # SQLite experiment logging
│   ├── interfaces.py          # Shared IRFBundle / model contract
│   ├── local_projections.py   # Local Projections IRFs
│   ├── model_adapter.py       # Unifies all models into one interface
│   ├── model_registry.py      # Model versioning/persistence
│   ├── sign_restriction_svar.py
│   ├── svar_cholesky.py
│   └── var_model.py
├── pipeline/
│   └── run_pipeline.py        # End-to-end orchestration, shock engine, scenario comparison
├── services/
│   └── pipeline_service.py    # Caching + run entry point
├── tests/
│   └── test_suite.py          # Full test coverage
├── utils/
│   ├── logger.py
│   ├── run_metadata.py
│   └── serialization.py       # JSON-safe serialization for all logging
├── main.py                    # CLI entry point
├── setup_env.py                # Venv creation + dependency install
├── requirements.txt
└── Makefile

---

## Next Steps / Extensions

- Calibration: Replace hardcoded betas with estimates from historical loan-level default data
- LGD Model: Add a Loss Given Default model for Expected Loss = PD x LGD x EAD
- Multi-horizon forecasting: Extend to 4/8-quarter horizons for DFAST-style multi-year paths
- CECL integration: Connect to an allowance-for-credit-losses calculation
- Portfolio view: Batch-score a loan book and aggregate to portfolio EL