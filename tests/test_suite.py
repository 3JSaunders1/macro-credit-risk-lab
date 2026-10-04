"""
tests/test_suite.py
-------------------
Full test suite covering all models, pipeline, and API.
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_macro_df():
    np.random.seed(42)
    n = 40
    u = np.clip(5.0 + np.cumsum(np.random.normal(0, 0.3, n)), 2, 20)
    pi = np.clip(2.5 + np.cumsum(np.random.normal(0, 0.2, n)), -2, 15)
    return pd.DataFrame({"unemployment": u, "inflation": pi})


# ── Credit Model ──────────────────────────────────────────────────────────────

class TestCreditModel:

    def setup_method(self):
        from models.credit_model import CreditModel
        self.model = CreditModel()

    def test_pd_is_probability(self):
        pd_val = self.model.predict_pd(5.0, 2.5)
        assert 0 < pd_val < 1

    def test_higher_unemployment_raises_pd(self):
        assert self.model.predict_pd(10.0, 2.5) > self.model.predict_pd(3.0, 2.5)

    def test_batch_prediction(self):
        df = pd.DataFrame({"unemployment": [4.0, 8.0, 12.0],
                            "inflation": [2.0, 4.0, 6.0]})
        pds = self.model.predict_pd_batch(df)
        assert len(pds) == 3
        assert all(0 < p < 1 for p in pds)

    def test_rating_valid(self):
        valid = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"}
        assert self.model.pd_to_rating(0.01) in valid

    def test_stress_test_returns_dataframe(self):
        from models.credit_model import StressScenario
        scenarios = [StressScenario("Test", 5.0, 2.5)]
        df = self.model.stress_test(scenarios)
        assert isinstance(df, pd.DataFrame)
        assert "predicted_pd" in df.columns

    def test_sensitivity_keys(self):
        sens = self.model.sensitivity()
        assert "base_pd" in sens
        assert "unemployment_shock_delta" in sens
        assert "inflation_shock_delta" in sens

    def test_unemployment_shock_positive(self):
        sens = self.model.sensitivity()
        assert sens["unemployment_shock_delta"] > 0


# ── VAR Model ─────────────────────────────────────────────────────────────────

class TestVARModel:

    def test_fit_forecast(self, sample_macro_df):
        from models.var_model import VARModel
        model = VARModel(lags=1).fit(sample_macro_df)
        fc = model.forecast(steps=1)
        assert fc.shape == (1, 2)

    def test_irf_bundle(self, sample_macro_df):
        from models.var_model import VARModel
        model = VARModel(lags=1).fit(sample_macro_df)
        bundle = model.irf(horizon=5)
        assert len(bundle.irfs) == 6
        assert bundle.fevd is not None

    def test_diagnostics(self, sample_macro_df):
        from models.var_model import VARModel
        model = VARModel(lags=1).fit(sample_macro_df)
        diag = model.diagnostics()
        assert "var_lags_used" in diag
        assert "var_stability" in diag


# ── SVAR Cholesky ─────────────────────────────────────────────────────────────

class TestSVARCholesky:

    def test_fit_forecast(self, sample_macro_df):
        from models.svar_cholesky import SVARCholesky
        model = SVARCholesky(lags=1).fit(sample_macro_df)
        fc = model.forecast(steps=1)
        assert fc.shape == (1, 2)

    def test_irf_bundle(self, sample_macro_df):
        from models.svar_cholesky import SVARCholesky
        model = SVARCholesky(lags=1).fit(sample_macro_df)
        bundle = model.irf(periods=4)
        assert len(bundle.irfs) == 5


# ── BVAR ─────────────────────────────────────────────────────────────────────

class TestBVAR:

    def test_fit_forecast(self, sample_macro_df):
        from models.bvar import BVARModel
        model = BVARModel(lags=1, lambda_=0.2).fit(sample_macro_df)
        fc = model.forecast(steps=1)
        assert fc.shape == (1, 2)

    def test_irf_bundle(self, sample_macro_df):
        from models.bvar import BVARModel
        model = BVARModel(lags=1).fit(sample_macro_df)
        bundle = model.irf(horizon=6)
        assert len(bundle.irfs) == 6
        assert bundle.fevd is not None

    def test_forecast_finite(self, sample_macro_df):
        from models.bvar import BVARModel
        model = BVARModel(lags=1).fit(sample_macro_df)
        fc = model.forecast(steps=4)
        assert np.all(np.isfinite(fc))


# ── Local Projections ─────────────────────────────────────────────────────────

class TestLocalProjections:

    def test_fit_irf(self, sample_macro_df):
        from models.local_projections import LocalProjections
        model = LocalProjections(horizon=6, lags=1).fit(sample_macro_df)
        bundle = model.irf()
        assert len(bundle.irfs) <= 6

    def test_forecast_shape(self, sample_macro_df):
        from models.local_projections import LocalProjections
        model = LocalProjections(horizon=4, lags=1).fit(sample_macro_df)
        fc = model.forecast(steps=1)
        assert fc.shape == (1, 2)


# ── Pipeline ─────────────────────────────────────────────────────────────────

class TestPipeline:

    def test_pipeline_keys(self):
        from pipeline.run_pipeline import run_pipeline
        result = run_pipeline(run_stress=False, var_lags=1)
        required = {"forecast", "forecast_ci", "predicted_pd", "rating",
                    "sensitivity", "stationarity", "var_lags_used", "irf"}
        assert required.issubset(result.keys())

    def test_pipeline_pd_valid(self):
        from pipeline.run_pipeline import run_pipeline
        result = run_pipeline(run_stress=False, var_lags=1)
        assert 0 < result["predicted_pd"] < 1

    def test_pipeline_rating_valid(self):
        from pipeline.run_pipeline import run_pipeline
        valid = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"}
        result = run_pipeline(run_stress=False, var_lags=1)
        assert result["rating"] in valid

    def test_pipeline_stationarity(self):
        from pipeline.run_pipeline import run_pipeline
        result = run_pipeline(run_stress=False, var_lags=1)
        assert "unemployment" in result["stationarity"]
        assert "inflation" in result["stationarity"]

    def test_pipeline_stress(self):
        from pipeline.run_pipeline import run_pipeline
        result = run_pipeline(run_stress=True, var_lags=1)
        assert result["stress_test"] is not None
        assert len(result["stress_test"]) > 0

    def test_pipeline_bvar(self):
        from pipeline.run_pipeline import run_pipeline
        result = run_pipeline(model_type="bvar", run_stress=False, var_lags=1)
        assert 0 < result["predicted_pd"] < 1

    def test_pipeline_lp(self):
        from pipeline.run_pipeline import run_pipeline
        result = run_pipeline(model_type="local_projections",
                               run_stress=False, var_lags=1)
        assert "predicted_pd" in result

# ── API ───────────────────────────────────────────────────────────────────────

class TestAPI:

    @pytest.fixture(autouse=True)
    def setup(self):
        from fastapi.testclient import TestClient
        from api.server import app
        self.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        assert r.status_code == 200

    def test_forecast_and_score(self):
        r = self.client.get("/forecast_and_score")
        assert r.status_code == 200
        assert "predicted_pd" in r.json()

    def test_score_endpoint(self):
        r = self.client.post("/score",
                              json={"unemployment": 8.0, "inflation": 4.0})
        assert r.status_code == 200
        data = r.json()
        assert 0 < data["predicted_pd"] < 1

    def test_score_invalid(self):
        r = self.client.post("/score",
                              json={"unemployment": -1, "inflation": 2.0})
        assert r.status_code == 422

    def test_stress_test_endpoint(self):
        r = self.client.get("/stress_test")
        assert r.status_code == 200
        assert len(r.json()) > 0