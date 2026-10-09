# pylint: disable=wrong-import-position
"""
models/loss_validation.py
-------------------------
Validate the dynamic loss model the way stress-testing models are judged:
  1. One-step-ahead out-of-time accuracy vs. naive persistence and the static model
  2. Conditional backtest: simulate 2008-2010 losses from the actual macro path
  3. COVID: the same simulation from the end of 2019

Usage: python -m models.loss_validation
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.loss_model import LossModel, load_data
from models.credit_estimation import fit_fractional_logit, predict

TRAIN_END = "2006-10-01"
BACKTESTS = {"2008 crisis": ("2007-10-01", "2010-10-01"),
             "COVID": ("2019-10-01", "2021-10-01")}


from utils.logging_utils import get_logger, run_main

log = get_logger(__name__)


def rmse_pp(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt((((predicted - actual) * 100) ** 2).mean()))


def main():
    from pathlib import Path
    fig_dir = Path(__file__).resolve().parents[1] / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train, test = df.loc[:TRAIN_END], df.loc[TRAIN_END:].iloc[1:]
    model = LossModel().fit(train)

    # --- 1. One-step-ahead, out of time ---
    persistence = test["loss_rate"].shift(1).fillna(train["loss_rate"].iloc[-1])
    static = fit_fractional_logit(train, ["unemployment", "inflation"])
    rows = [
        {"model": "Naive persistence (last quarter)", "rmse_pp": rmse_pp(test["loss_rate"], persistence)},
        {"model": "Static levels model", "rmse_pp": rmse_pp(test["loss_rate"],
                                                             predict(static, test, ["unemployment", "inflation"]))},
        {"model": "Dynamic loss model", "rmse_pp": rmse_pp(test["loss_rate"], model.predict_one_step(test))},
    ]
    print(f"Trained on {train.index.min().date()} to {train.index.max().date()} ({len(train)} quarters)")
    print("\n=== 1. One-step-ahead accuracy, 2007 onward (RMSE, percentage points) ===")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== Coefficients (training 1985-2006) ===")
    r = model.result
    print(pd.DataFrame({"coef": r.params, "se_hac": r.bse, "p_value": r.pvalues}).round(4).to_string())
    rho = model.params["lag_logit_loss"]
    print(f"Persistence rho = {rho:.3f}; half-life of a shock ~ {np.log(0.5) / np.log(rho):.1f} quarters")

    # --- 2 and 3. Conditional backtests ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (label, (start, end)) in zip(axes, BACKTESTS.items()):
        start_rate = df.loc[start, "loss_rate"]
        path = df.loc[start:end].iloc[1:]
        sim = model.simulate(start_rate, path)
        actual = path["loss_rate"]
        print(f"\n=== {label}: conditional backtest from {start[:7]} (actual macro path) ===")
        print(f"Start {start_rate:.2%} | actual peak {actual.max():.2%} | simulated peak {sim.max():.2%} "
              f"| cumulative loss ratio (sim/actual) {sim.sum() / actual.sum():.2f}")

        ax.plot(actual.index, actual * 100, "k-o", label="Actual")
        ax.plot(sim.index, sim * 100, "s--", label="Simulated (dynamic model)")
        ax.set_title(f"{label}: conditional backtest"); ax.set_ylabel("Charge-off rate (%)")
        ax.legend(); ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(fig_dir / "loss_model_backtests.png", dpi=150); plt.close()

    # --- Production model: full sample ---
    full = LossModel().fit(df)
    full.save(sample=f"{df.index.min().date()} to {df.index.max().date()}")
    print("\n=== Full-sample coefficients (saved for the scenario engine) ===")
    r = full.result
    print(pd.DataFrame({"coef": r.params, "se_hac": r.bse, "p_value": r.pvalues}).round(4).to_string())
    log.info(f"Saved parameters to config/loss_model.json and chart to {fig_dir}")
if __name__ == "__main__":
    run_main(main)
