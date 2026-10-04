# pylint: disable=wrong-import-position
"""
pipeline/scenario_engine.py
---------------------------
Model-based stress scenarios.

1. Baseline: the BVAR's forecast over the scenario horizon.
2. Scenario paths: baseline + responses to structural (Cholesky) shocks, propagated
   through the BVAR's impulse responses and scaled to each scenario's target peak.
3. Losses: the dynamic loss model, simulated forward from the latest actual
   charge-off rate along each path (no pandemic-style policy support assumed).

Usage: python -m pipeline.scenario_engine
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.scenarios import CUMULATIVE_QUARTERS, HORIZON, SCENARIOS
from models.bvar import BVARModel
from models.loss_model import LossModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "macro_data.csv"
FIG_DIR = ROOT / "reports" / "figures"
VARS = ["unemployment", "inflation"]
REFERENCE_2008 = ("2008-01-01", "2010-01-01")      # 9 quarters of actual losses


def load_history() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
    df.index.freq = "QS"
    return df


def shock_response(irfs: np.ndarray, shock_idx: int, quarters: int, horizon: int) -> np.ndarray:
    """Response of all variables to `quarters` consecutive unit shocks; shape (horizon, n)."""
    unit = np.zeros((horizon, irfs.shape[1]))
    for s in range(quarters):
        for h in range(s, horizon):
            unit[h] += irfs[h - s, :, shock_idx]
    return unit


def scale_to_peak(unit: np.ndarray, var_idx: int, peak: float) -> np.ndarray:
    """Scale a response so the shocked variable's maximum deviation equals `peak`."""
    max_dev = unit[:, var_idx].max()
    if max_dev <= 0:
        raise ValueError("Shock does not raise the target variable; cannot scale to a peak.")
    return unit * (peak / max_dev)


def build_path(model: BVARModel, last_date: pd.Timestamp, horizon: int, shocks: dict) -> pd.DataFrame:
    """Baseline forecast plus scaled structural shock responses."""
    baseline = model.forecast(horizon)
    irfs = np.array(model.irf(horizon).irfs)
    deviation = np.zeros_like(baseline)
    for var, spec in shocks.items():
        j = VARS.index(var)
        unit = shock_response(irfs, j, spec["quarters"], horizon)
        deviation += scale_to_peak(unit, j, spec["peak"])
    idx = pd.date_range(last_date + pd.DateOffset(months=3), periods=horizon, freq="QS")
    return pd.DataFrame(baseline + deviation, index=idx, columns=VARS)


def loss_path(loss_model: LossModel, history: pd.DataFrame, path: pd.DataFrame) -> pd.Series:
    """Simulate losses along a scenario path, starting from the latest actual charge-off rate."""
    u = pd.concat([history["unemployment"].iloc[-4:], path["unemployment"]])
    inputs = pd.DataFrame({"d4_unemployment": (u - u.shift(4)).loc[path.index],
                           "inflation": path["inflation"]}, index=path.index)
    start = history["charge_off_rate"].dropna().iloc[-1] / 100
    return loss_model.simulate(start, inputs)


def cumulative_loss(rates: pd.Series, quarters: int = CUMULATIVE_QUARTERS) -> float:
    """Cumulative loss as a share of balances: annualized quarterly rates / 4, summed."""
    return float(rates.iloc[:quarters].sum() / 4)


def run_scenarios(lags: int = 2, lambda_: float = 0.2):
    history = load_history()
    model = BVARModel(lags=lags, lambda_=lambda_).fit(history[VARS])
    loss_model = LossModel.load()
    results = {}
    for name, spec in SCENARIOS.items():
        path = build_path(model, history.index[-1], HORIZON, spec["shocks"])
        path["loss_rate"] = loss_path(loss_model, history, path)
        results[name] = path
    return results, history


def main():
    results, history = run_scenarios()
    start = history["charge_off_rate"].dropna().iloc[-1]
    ref = history.loc[REFERENCE_2008[0]:REFERENCE_2008[1], "charge_off_rate"] / 100

    rows = []
    for name, path in results.items():
        rows.append({"scenario": name,
                     "peak_unemployment": path["unemployment"].max(),
                     "peak_inflation": path["inflation"].max(),
                     "peak_loss_rate_pct": path["loss_rate"].max() * 100,
                     "cumulative_9q_loss_pct": cumulative_loss(path["loss_rate"]) * 100})
    summary = pd.DataFrame(rows)

    print(f"Scenarios start {results['Baseline'].index[0].date()}; "
          f"latest actual charge-off rate {start:.2f}%")
    print("\n=== Scenario results ===")
    print(summary.round(2).to_string(index=False))
    print(f"\nReference: actual 2008 Q1-2010 Q1 cumulative loss = {cumulative_loss(ref) * 100:.2f}% "
          f"(peak quarterly rate {ref.max() * 100:.2f}%)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(FIG_DIR / "scenario_results.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, path in results.items():
        axes[0].plot(path.index, path["unemployment"], label=name)
        axes[1].plot(path.index, path["inflation"], label=name)
        axes[2].plot(path.index, path["loss_rate"] * 100, label=name)
    for ax, title in zip(axes, ["Unemployment (%)", "Inflation (%)", "Charge-off rate (%)"]):
        ax.set_title(title); ax.tick_params(axis="x", rotation=45)
    axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "scenario_paths.png", dpi=150); plt.close()
    print(f"\nSaved results and chart to {FIG_DIR}")


if __name__ == "__main__":
    main()
