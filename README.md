# Macro-Driven Credit Scoring Model

A production-structured Python project that chains a **Vector Autoregression (VAR)** macro forecasting model with a **logistic Probability of Default (PD)** model. The pipeline forecasts unemployment and inflation one quarter ahead, then uses those forecasts to estimate credit risk.

---

## Architecture

```
data/macro_data.csv
       │
       ▼
  [VAR Model]  ──── fits on (unemployment, inflation) time series
       │             selects lag order via AIC
       │             forecasts 1 step ahead w/ 90% CI
       ▼
 [Credit Model] ─── logistic PD: PD = σ(β₀ + β₁·u + β₂·π)
       │              maps PD → internal rating (AAA → D)
       │              runs stress scenarios (Baseline, Adverse, etc.)
       ▼
  FastAPI / CLI
```

---

## Quick Start

### Option 1: Local Python

```bash
# 1. Create virtual environment and install dependencies
python setup_env.py

# 2. Activate the environment
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Run the pipeline
python main.py

# 4. Start the API server
uvicorn api.server:app --reload --port 8000
```

### Option 2: Makefile (recommended)

```bash
make setup        # create venv + install
make run          # run CLI pipeline
make api          # start API with hot-reload
make test         # run all tests
```

### Option 3: Docker

```bash
# API server
docker build -t credit-model .
docker run -p 8000:8000 credit-model

# CLI
docker run --rm credit-model python main.py

# Or use docker-compose
docker compose up --build
```

---

## API Endpoints

Once the server is running at `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/forecast_and_score` | Full pipeline: VAR forecast → PD score |
| POST | `/score` | Score a custom macro scenario |
| GET | `/stress_test` | Standard DFAST-style scenarios |
| POST | `/stress_test/custom` | User-defined scenario grid |

Interactive docs: `http://localhost:8000/docs`

### Example: Score a custom scenario

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"unemployment": 8.5, "inflation": 4.0}'
```

```json
{
  "unemployment": 8.5,
  "inflation": 4.0,
  "predicted_pd": 0.074821,
  "rating": "BB"
}
```

---

## Models

### VAR Model (`models/var_model.py`)

- Fits a VAR(p) on unemployment and inflation
- Lag order p selected automatically via AIC (or set manually)
- Includes ADF stationarity testing
- Returns point forecasts + 90% confidence intervals

### Credit Model (`models/credit_model.py`)

- Logistic regression: `PD = σ(β₀ + β₁·unemployment + β₂·inflation)`
- Default coefficients set to economically sensible priors
- `fit()` method: estimate from historical default data via sklearn
- `stress_test()`: runs Baseline / Adverse / Severely Adverse / Stagflation / COVID-like scenarios
- `sensitivity()`: marginal PD impact of 1pp macro shocks
- `pd_to_rating()`: maps PD to simplified Basel-style rating scale

---

## Stress Scenarios

| Scenario | Unemployment | Inflation |
|----------|-------------|-----------|
| Baseline | 4.0% | 2.5% |
| Adverse | 7.0% | 4.0% |
| Severely Adverse | 12.0% | 1.5% |
| Stagflation | 8.0% | 9.0% |
| COVID-like Shock | 15.0% | 0.5% |

Loosely inspired by Federal Reserve DFAST scenario design.

---

## Testing

```bash
pytest tests/ -v          # full test suite
pytest tests/ -x          # stop on first failure
pytest tests/ --cov=.     # with coverage
```

Tests cover: credit model math, VAR fit/forecast, full pipeline integration, and all API endpoints.

---

## Project Structure

```
credit_modeling/
├── api/
│   └── server.py          # FastAPI app
├── data/
│   └── macro_data.csv     # Quarterly macro data (2010–2024)
├── models/
│   ├── credit_model.py    # Logistic PD model + stress testing
│   └── var_model.py       # VAR macro forecasting model
├── pipeline/
│   └── run_pipeline.py    # End-to-end orchestration
├── tests/
│   └── test_suite.py      # Full test coverage
├── main.py                # CLI entry point
├── setup_env.py           # Venv creation + dependency install
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## Next Steps / Extensions

- **Calibration**: Replace hardcoded betas with estimates from historical loan-level default data
- **LGD Model**: Add a Loss Given Default model for Expected Loss = PD × LGD × EAD
- **Multi-horizon forecasting**: Extend VAR to 4/8-quarter horizons for DFAST-style multi-year paths
- **CECL integration**: Connect to an allowance-for-credit-losses calculation
- **Portfolio view**: Batch-score a loan book and aggregate to portfolio EL
- **Forecasting project integration**: Share the VAR macro module across the forecasting project
