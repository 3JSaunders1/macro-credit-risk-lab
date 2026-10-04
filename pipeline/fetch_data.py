"""
Fetch reproducible quarterly macro and credit data from FRED.

Builds data/macro_data.csv with:
  - unemployment:    quarterly average of the monthly civilian unemployment rate (UNRATE)
  - inflation:       year-over-year % change in the quarterly average CPI (CPIAUCSL)
  - charge_off_rate: charge-off rate on consumer loans, all commercial banks (CORCACBS)
and writes data/macro_data_meta.json documenting sources and transformations.

Usage: python -m pipeline.fetch_data
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "macro_data.csv"
OUT_META = ROOT / "data" / "macro_data_meta.json"
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
START = "1985-01-01"

SERIES = {
    "UNRATE": "Civilian unemployment rate, monthly, seasonally adjusted",
    "CPIAUCSL": "CPI for all urban consumers, all items, monthly, seasonally adjusted",
    "CORCACBS": "Charge-off rate on consumer loans, all commercial banks, quarterly, SA",
}


def fetch_series(series_id: str, start: str, api_key: str) -> pd.Series:
    params = {"series_id": series_id, "api_key": api_key,
              "file_type": "json", "observation_start": start}
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = pd.DataFrame(resp.json()["observations"])
    values = pd.to_numeric(obs["value"], errors="coerce")   # FRED uses "." for missing
    return pd.Series(values.to_numpy(), index=pd.to_datetime(obs["date"]), name=series_id)


def fill_interior_gaps(s: pd.Series) -> tuple[pd.Series, list[str]]:
    """Linearly interpolate missing months between known values (e.g., data not
    collected during a government shutdown). Trailing months are never filled."""
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    missing = full[full.isna()].index.strftime("%Y-%m").tolist()
    return full.interpolate(limit_area="inside"), missing


def to_quarterly_mean(s: pd.Series, min_obs: int = 1) -> pd.Series:
    """Average observations into calendar quarters, keeping only complete quarters.

    min_obs: observations required per quarter (3 for monthly data, 1 for quarterly).
    """
    grouped = s.resample("QS")
    means, counts = grouped.mean(), grouped.count()
    return means.where(counts >= min_obs)


def yoy_pct(s: pd.Series, periods: int = 4) -> pd.Series:
    """Year-over-year % change for a quarterly series."""
    return (s / s.shift(periods) - 1) * 100


def build_dataset(raw: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict]:
    unrate, unrate_filled = fill_interior_gaps(raw["UNRATE"])
    cpi, cpi_filled = fill_interior_gaps(raw["CPIAUCSL"])

    unemployment = to_quarterly_mean(unrate, min_obs=3)
    inflation = yoy_pct(to_quarterly_mean(cpi, min_obs=3))
    charge_off = to_quarterly_mean(raw["CORCACBS"], min_obs=1)

    df = pd.concat({"unemployment": unemployment, "inflation": inflation,
                    "charge_off_rate": charge_off}, axis=1, sort=True)
    df = df.loc[START:].dropna(subset=["unemployment", "inflation"])
    df.index.name = "date"

    expected = pd.date_range(df.index.min(), df.index.max(), freq="QS")
    if len(expected) != len(df):
        raise ValueError("Quarterly series has gaps; time-series models need consecutive quarters.")

    return df.round(3), {"UNRATE": unrate_filled, "CPIAUCSL": cpi_filled}


def main():
    load_dotenv()
    api_key = os.environ["FRED_API_KEY"]

    # CPI starts a year early so year-over-year inflation exists from START
    cpi_start = (pd.Timestamp(START) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    raw = {
        "UNRATE": fetch_series("UNRATE", START, api_key),
        "CPIAUCSL": fetch_series("CPIAUCSL", cpi_start, api_key),
        "CORCACBS": fetch_series("CORCACBS", START, api_key),
    }
    df, filled = build_dataset(raw)
    df.to_csv(OUT_CSV)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "FRED, Federal Reserve Bank of St. Louis",
        "series": SERIES,
        "transformations": {
            "unemployment": "quarterly average of monthly UNRATE (complete quarters only)",
            "inflation": "year-over-year % change of quarterly average CPIAUCSL (complete quarters only)",
            "charge_off_rate": "CORCACBS, quarterly",
            "missing_months": "interior missing months linearly interpolated; trailing incomplete quarters dropped",
        },
        "interpolated_months": filled,
        "frequency": "quarterly (quarter-start dates)",
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "rows": len(df),
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    print(f"Saved {len(df)} quarters ({meta['start']} to {meta['end']}) to {OUT_CSV}")
    print(f"Charge-off rate available for {df['charge_off_rate'].notna().sum()} quarters")
    print(f"Interpolated months: {filled}")
    print(df.tail(4).to_string())


if __name__ == "__main__":
    main()