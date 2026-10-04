# pylint: disable=wrong-import-position
"""
models/credit_estimation.py
---------------------------
Estimate the macro-to-credit link on real consumer charge-off rates with a
fractional logit, validated out of time through the 2008 crisis and COVID.

Usage: python -m models.credit_estimation
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "macro_data.csv"
COEF_FILE = ROOT / "config" / "static_model_coefficients.json"   # documentation only; not loaded by CreditModel
FIG_DIR = ROOT / "reports" / "figures"
TRAIN_END = "2006-10-01"   # last training quarter; out-of-time test starts in 2007
SPECS = {
    "contemporaneous": ["unemployment", "inflation"],
    "unemployment lagged 2q": ["unemployment_lag2", "inflation"],
}
PERIODS = {"2008 crisis (2008-2010)": ("2008-01-01", "2010-10-01"),
           "COVID (2020-2021)": ("2020-01-01", "2021-10-01")}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date")
    df["loss_rate"] = df["charge_off_rate"] / 100          # percent -> fraction
    df["unemployment_lag2"] = df["unemployment"].shift(2)
    return df.dropna(subset=["loss_rate", "unemployment_lag2"])


def fit_fractional_logit(df: pd.DataFrame, features: list[str]):
    """Fractional logit: binomial GLM with a logit link on a rate between 0 and 1,
    with HAC standard errors for serial correlation."""
    X = sm.add_constant(df[features])
    return sm.GLM(df["loss_rate"], X, family=sm.families.Binomial()).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4})


def predict(result, df: pd.DataFrame, features: list[str]) -> pd.Series:
    X = sm.add_constant(df[features], has_constant="add")
    return pd.Series(result.predict(X), index=df.index)


def error_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    err = (predicted - actual) * 100                       # in percentage points
    return {"rmse_pp": float(np.sqrt((err ** 2).mean())),
            "mae_pp": float(err.abs().mean()),
            "bias_pp": float(err.mean())}


def main():
    df = load_data()
    train, test = df.loc[:TRAIN_END], df.loc[TRAIN_END:].iloc[1:]
    print(f"Train: {train.index.min().date()} to {train.index.max().date()} ({len(train)} quarters)")
    print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} quarters)")

    rows, fits = [], {}
    benchmark = pd.Series(train["loss_rate"].mean(), index=test.index)
    rows.append({"model": "Benchmark: training mean", **error_metrics(test["loss_rate"], benchmark)})

    for name, feats in SPECS.items():
        res = fit_fractional_logit(train, feats)
        fits[name] = (res, feats)
        rows.append({"model": f"Fractional logit ({name})",
                     **error_metrics(test["loss_rate"], predict(res, test, feats))})

    results = pd.DataFrame(rows)
    print("\n=== Out-of-time accuracy, 2007 onward (percentage points of charge-off rate) ===")
    print(results.round(3).to_string(index=False))

    # Errors in stress periods for the contemporaneous model
    res, feats = fits["contemporaneous"]
    pred_all = predict(res, df, feats)
    print("\n=== Contemporaneous model in stress periods ===")
    for label, (start, end) in PERIODS.items():
        window = df.loc[start:end]
        m = error_metrics(window["loss_rate"], pred_all.loc[start:end])
        print(f"{label}: actual avg {window['loss_rate'].mean():.2%}, "
              f"predicted avg {pred_all.loc[start:end].mean():.2%}, bias {m['bias_pp']:+.2f} pp")

    # Coefficients: training sample, then full sample for production use
    print("\n=== Coefficients, contemporaneous model (training 1985-2006) ===")
    print(pd.DataFrame({"coef": res.params, "se_hac": res.bse,
                        "p_value": res.pvalues}).round(4).to_string())

    full = fit_fractional_logit(df, SPECS["contemporaneous"])
    print("\n=== Coefficients, full sample (documentation only; not used by the pipeline) ===")
    print(pd.DataFrame({"coef": full.params, "se_hac": full.bse,
                        "p_value": full.pvalues}).round(4).to_string())

    COEF_FILE.write_text(json.dumps({
        "beta_0": float(full.params["const"]),
        "beta_u": float(full.params["unemployment"]),
        "beta_pi": float(full.params["inflation"]),
        "target": "consumer loan charge-off rate, all commercial banks (CORCACBS), as a fraction",
        "model": "fractional logit (binomial GLM, logit link), HAC standard errors",
        "sample": f"{df.index.min().date()} to {df.index.max().date()}",
        "estimated_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2))

    # Chart: actual vs. model, training fit and out-of-time prediction
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df.index, df["loss_rate"] * 100, color="black", label="Actual charge-off rate")
    ax.plot(pred_all.loc[:TRAIN_END].index, pred_all.loc[:TRAIN_END] * 100, "--",
            label="Model (training fit)")
    ax.plot(pred_all.loc[test.index].index, pred_all.loc[test.index] * 100,
            label="Model (out-of-time)")
    for start, end in PERIODS.values():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey", alpha=0.15)
    ax.axvline(pd.Timestamp(TRAIN_END), color="red", linestyle=":", label="Train/test split")
    ax.set_ylabel("Charge-off rate (%)"); ax.legend(loc="upper left")
    ax.set_title("Consumer charge-off rate: actual vs. macro model")
    plt.tight_layout(); plt.savefig(FIG_DIR / "credit_link_oot.png", dpi=150); plt.close()

    results.to_csv(FIG_DIR / "credit_link_oot_metrics.csv", index=False)
    print(f"\nSaved coefficients to {COEF_FILE} and chart to {FIG_DIR}")


if __name__ == "__main__":
    main()
