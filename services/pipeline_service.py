"""
services/pipeline_service.py
----------------------------
Single entry point for all pipeline execution.
Handles caching, scenario comparison, and run logging.
"""

from pipeline.run_pipeline import run_pipeline, run_scenario_comparison
from utils.run_metadata import save_run_metadata

_cached_result = None


def get_pipeline_result(force_refresh: bool = False):
    global _cached_result
    if _cached_result is None or force_refresh:
        _cached_result = run_pipeline()
        try:
            save_run_metadata(_cached_result)
        except Exception:
            pass
    return _cached_result


def run_fresh_pipeline(**kwargs):
    result = run_pipeline(**kwargs)
    try:
        save_run_metadata(result)
    except Exception:
        pass
    return result


def run_comparison(scenarios: list, model_type: str = "var", var_lags: int = 2):
    return run_scenario_comparison(
        scenarios=scenarios,
        model_type=model_type,
        var_lags=var_lags,
    )