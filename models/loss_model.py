"""
models/loss_model.py
--------------------
Dynamic credit loss model for stress testing.

    logit(loss_t) = a + rho * logit(loss_{t-1}) + b * d4_unemployment_t
                    + c * inflation_t + event indicators

Estimated by fractional logit on the consumer charge-off rate. Because losses
depend on last quarter's losses, the model can simulate a loss path forward
from a starting point, given a macro scenario path.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "macro_data.csv"
PARAM_FILE = ROOT / "config" / "loss_model.json"

FEATURES = ["lag_logit_loss", "d4_unemployment", "inflation",
            "bankruptcy_2005q4", "bankruptcy_2006q1", "covid_policy"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Build model variables from the quarterly macro data."""
    df = df.copy()
    df["loss_rate"] = df["charge_off_rate"] / 100
    df["logit_loss"] = logit(df["loss_rate"])
    df["lag_logit_loss"] = df["logit_loss"].shift(1)
    df["d4_unemployment"] = df["unemployment"] - df["unemployment"].shift(4)
    df["bankruptcy_2005q4"] = (df.index == "2005-10-01").astype(int)   # filings rush before the 2005 law
    df["bankruptcy_2006q1"] = (df.index == "2006-01-01").astype(int)   # drop after the rush
    df["covid_policy"] = ((df.index >= "2020-04-01")
                          & (df.index <= "2021-10-01")).astype(int)    # stimulus and forbearance
    return df.dropna(subset=["loss_rate", "lag_logit_loss", "d4_unemployment"])


def load_data() -> pd.DataFrame:
    return prepare(pd.read_csv(DATA, parse_dates=["date"]).set_index("date"))


class LossModel:
    def __init__(self, features: list[str] | None = None):
        self.features = features or FEATURES
        self.used: list[str] = []
        self.params: pd.Series | None = None
        self.result = None

    def fit(self, df: pd.DataFrame) -> "LossModel":
        # Drop indicators that are all zero in this sample (e.g., COVID before 2007)
        self.used = [f for f in self.features if df[f].abs().sum() > 0]
        X = sm.add_constant(df[self.used])
        self.result = sm.GLM(df["loss_rate"], X, family=sm.families.Binomial()).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4})
        self.params = self.result.params
        return self

    def predict_one_step(self, df: pd.DataFrame) -> pd.Series:
        """Predict each quarter using the previous quarter's actual losses."""
        X = sm.add_constant(df[self.used], has_constant="add")
        return pd.Series(self.result.predict(X), index=df.index)

    def simulate(self, start_rate: float, path: pd.DataFrame) -> pd.Series:
        """Simulate losses forward from start_rate along a macro path.

        path: one row per quarter with 'd4_unemployment' and 'inflation'
        (event indicators optional, default 0). Each quarter's predicted loss
        becomes the next quarter's lagged loss.
        """
        rate, out = start_rate, []
        for _, row in path.iterrows():
            z = self.params["const"] + self.params["lag_logit_loss"] * logit(rate)
            for f in self.used:
                if f != "lag_logit_loss":
                    z += self.params[f] * float(row.get(f, 0.0))
            rate = float(expit(z))
            out.append(rate)
        return pd.Series(out, index=path.index)

    def save(self, path: Path = PARAM_FILE, sample: str = "") -> None:
        path.write_text(json.dumps({
            "params": {k: float(v) for k, v in self.params.items()},
            "features": self.used,
            "target": "consumer loan charge-off rate (CORCACBS), as a fraction",
            "model": "dynamic fractional logit, HAC standard errors",
            "sample": sample,
        }, indent=2))

    @classmethod
    def load(cls, path: Path = PARAM_FILE) -> "LossModel":
        spec = json.loads(path.read_text())
        model = cls(spec["features"])
        model.used = spec["features"]
        model.params = pd.Series(spec["params"])
        return model
