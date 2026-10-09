# pylint: disable=wrong-import-position
"""
analysis/forecast_backtest.py
-----------------------------
Rolling out-of-sample forecast backtest: VAR and BVAR vs. a random walk.

At each origin quarter, models are fit only on data up to that quarter and
forecast 1 and 4 quarters ahead. Reports RMSE, relative RMSE vs. the random walk,
Diebold-Mariano tests, and 90% interval coverage, with and without COVID.

The two SVARs share the VAR's reduced-form forecasts (identification changes
shock interpretation, not forecasts), and local projections forecast as a random
walk, so three distinct forecasters are compared.

Usage: python -m analysis.forecast_backtest
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from models.bvar import BVARModel
from models.local_projections import LocalProjections
from models.var_model import VARModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "macro_data.csv"
FIG_DIR = ROOT / "reports" / "figures"
VARS = ["unemployment", "inflation"]
FIRST_ORIGIN = "2000-01-01"
HORIZONS = [1, 4]
ALPHA = 0.10
COVID = ("2020-01-01", "2021-10-01")      # target quarters excluded in the "ex-COVID" results
MODELS = {
    "VAR(2)": lambda: VARModel(lags=2),
    "BVAR(2)": lambda: BVARModel(lags=2, lambda_=0.2),
    "Random walk": lambda: LocalProjections(horizon=1, lags=1),   # forecasts as a random walk
}


from utils.logging_utils import get_logger, run_main

log = get_logger(__name__)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")[VARS]
    df.index.freq = "QS"          # quarterly, quarter-start dates
    return df


def rolling_forecasts(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """One row per (model, horizon, origin, variable), using only data up to each origin."""
    rows, max_h = [], max(horizons)
    origins = df.index[(df.index >= FIRST_ORIGIN)][: -1]
    for origin in origins:
        train = df.loc[:origin]
        for name, make in MODELS.items():
            point, lower, upper = make().fit(train).forecast_interval(steps=max_h, alpha=ALPHA)
            for h in horizons:
                pos = df.index.get_loc(origin) + h
                if pos >= len(df):
                    continue
                target = df.index[pos]
                for j, var in enumerate(VARS):
                    rows.append({"model": name, "h": h, "origin": origin, "target": target,
                                 "variable": var, "forecast": point[h - 1, j],
                                 "lower": lower[h - 1, j], "upper": upper[h - 1, j],
                                 "actual": df.loc[target, var]})
    out = pd.DataFrame(rows)
    out["error"] = out["forecast"] - out["actual"]
    out["covered"] = (out["actual"] >= out["lower"]) & (out["actual"] <= out["upper"])
    return out


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int) -> tuple[float, float]:
    """DM test of equal squared-error accuracy, with a Newey-West variance (h-1 lags).
    Negative statistic: model 1 is more accurate."""
    d = e1 ** 2 - e2 ** 2
    T, dbar = len(d), d.mean()
    dc = d - dbar
    var = dc @ dc / T
    for k in range(1, h):
        var += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / T
    stat = dbar / np.sqrt(max(var, 1e-12) / T)
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def summarize(fc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, var), g in fc.groupby(["h", "variable"]):
        rw = g[g["model"] == "Random walk"].set_index("target")["error"]
        rw_rmse = np.sqrt((rw ** 2).mean())
        for name, m in g.groupby("model"):
            e = m.set_index("target")["error"]
            rmse = np.sqrt((e ** 2).mean())
            row = {"h": h, "variable": var, "model": name, "rmse": rmse,
                   "rel_rmse_vs_rw": rmse / rw_rmse, "coverage_90": m["covered"].mean()}
            if name != "Random walk":
                common = e.index.intersection(rw.index)
                row["dm_stat"], row["dm_p"] = diebold_mariano(
                    e.loc[common].to_numpy(), rw.loc[common].to_numpy(), h)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = load_data()
    print(f"Data: {df.index.min().date()} to {df.index.max().date()}; "
          f"origins from {FIRST_ORIGIN[:4]}; horizons {HORIZONS}")
    fc = rolling_forecasts(df)

    ex_covid = fc[~fc["target"].between(pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1]))]
    full_table, ex_table = summarize(fc), summarize(ex_covid)

    pd.set_option("display.width", 160)
    print("\n=== Full sample (2000 onward, including COVID) ===")
    print(full_table.round(3).to_string(index=False))
    print("\n=== Excluding COVID targets (2020-2021) ===")
    print(ex_table.round(3).to_string(index=False))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    full_table.to_csv(FIG_DIR / "forecast_backtest_full.csv", index=False)
    ex_table.to_csv(FIG_DIR / "forecast_backtest_ex_covid.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, var in zip(axes, VARS):
        sub = fc[(fc["h"] == 4) & (fc["variable"] == var)]
        actual = sub.drop_duplicates("target").set_index("target")["actual"]
        ax.plot(actual.index, actual, "k-", label="Actual")
        for name in ["VAR(2)", "BVAR(2)"]:
            s = sub[sub["model"] == name].set_index("target")["forecast"]
            ax.plot(s.index, s, label=f"{name}, 4q ahead")
        ax.set_title(f"{var.capitalize()}: 4-quarter-ahead forecasts"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "forecast_backtest.png", dpi=150); plt.close()
    log.info(f"Saved tables and chart to {FIG_DIR}")
if __name__ == "__main__":
    run_main(main)
