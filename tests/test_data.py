"""Data transformations: gap filling, quarterly averaging, and year-over-year inflation."""
import numpy as np
import pandas as pd
import pytest
from pipeline.fetch_data import to_quarterly_mean, yoy_pct, build_dataset, fill_interior_gaps


def test_quarterly_mean_averages_months():
    s = pd.Series([3.0, 4.0, 5.0, 6.0, 6.0, 6.0],
                  index=pd.date_range("2020-01-01", periods=6, freq="MS"))
    q = to_quarterly_mean(s)
    assert q.loc["2020-01-01"] == pytest.approx(4.0)
    assert q.loc["2020-04-01"] == pytest.approx(6.0)


def test_yoy_pct():
    s = pd.Series([100.0, 101.0, 102.0, 103.0, 110.0],
                  index=pd.date_range("2019-01-01", periods=5, freq="QS"))
    assert yoy_pct(s).iloc[-1] == pytest.approx(10.0)
    assert np.isnan(yoy_pct(s).iloc[0])


def test_build_dataset_columns_and_dates():
    months = pd.date_range("2023-01-01", periods=36, freq="MS")
    quarters = pd.date_range("2023-01-01", periods=12, freq="QS")
    raw = {
        "UNRATE": pd.Series(4.0, index=months),
        "CPIAUCSL": pd.Series(np.linspace(300, 330, 36), index=months),
        "CORCACBS": pd.Series(2.0, index=quarters),
    }
    df, _ = build_dataset(raw)
    assert list(df.columns) == ["unemployment", "inflation", "charge_off_rate"]
    assert df["inflation"].notna().all()
    assert (df.index.month.isin([1, 4, 7, 10])).all()


def test_incomplete_quarters_dropped():
    s = pd.Series([4.0, 4.0, 4.0, 5.0, 5.0],   # Q2 has only two months
                  index=pd.date_range("2020-01-01", periods=5, freq="MS"))
    q = to_quarterly_mean(s, min_obs=3)
    assert q.loc["2020-01-01"] == pytest.approx(4.0)
    assert np.isnan(q.loc["2020-04-01"])


def test_interior_gap_is_filled_but_trailing_is_not():
    s = pd.Series([1.0, 3.0, 5.0],
                  index=pd.to_datetime(["2025-09-01", "2025-11-01", "2025-12-01"]))
    filled, missing = fill_interior_gaps(s)
    assert missing == ["2025-10"]
    assert filled.loc["2025-10-01"] == pytest.approx(2.0)


def test_build_dataset_has_consecutive_quarters():
    months = pd.date_range("2023-01-01", periods=36, freq="MS").delete(10)   # one missing month
    quarters = pd.date_range("2023-01-01", periods=12, freq="QS")
    raw = {
        "UNRATE": pd.Series(4.0, index=months),
        "CPIAUCSL": pd.Series(np.linspace(300, 330, 35), index=months),
        "CORCACBS": pd.Series(2.0, index=quarters),
    }
    df, filled = build_dataset(raw)
    assert filled["UNRATE"] == ["2023-11"]
    assert len(df) == len(pd.date_range(df.index.min(), df.index.max(), freq="QS"))