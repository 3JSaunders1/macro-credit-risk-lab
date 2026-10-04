"""Scenario engine: shock construction, scaling, and loss ordering."""
import numpy as np
import pytest
from config.scenarios import CUMULATIVE_QUARTERS, HORIZON, SCENARIOS
from pipeline.scenario_engine import (cumulative_loss, run_scenarios,
                                      scale_to_peak, shock_response)


def test_shock_response_accumulates_consecutive_shocks():
    irfs = np.ones((5, 2, 2))
    unit = shock_response(irfs, shock_idx=0, quarters=2, horizon=5)
    assert unit[0, 0] == 1 and unit[1, 0] == 2 and unit[4, 0] == 2


def test_scale_to_peak_hits_target():
    unit = np.array([[0.2, 0.0], [0.5, 0.1], [0.3, 0.0]])
    assert scale_to_peak(unit, 0, 6.0)[:, 0].max() == pytest.approx(6.0)


def test_cumulative_loss():
    import pandas as pd
    assert cumulative_loss(pd.Series([0.04] * 12)) == pytest.approx(0.09)


def test_scenarios_are_well_defined():
    assert SCENARIOS["Baseline"]["shocks"] == {}
    assert HORIZON >= CUMULATIVE_QUARTERS


@pytest.fixture(scope="module")
def results():
    return run_scenarios()[0]


def test_unemployment_peak_matches_target(results):
    dev = results["Severely Adverse"]["unemployment"] - results["Baseline"]["unemployment"]
    assert dev.max() == pytest.approx(6.0, abs=1e-6)


def test_losses_rise_with_severity(results):
    base = cumulative_loss(results["Baseline"]["loss_rate"])
    adverse = cumulative_loss(results["Adverse"]["loss_rate"])
    severe = cumulative_loss(results["Severely Adverse"]["loss_rate"])
    assert base < adverse < severe
