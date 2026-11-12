"""Unit tests for the data layer's universe filters."""
from __future__ import annotations

import pandas as pd
import pytest

from bimodality.data import _listing_age_mask


def _reference_loop(dates, tickers, list_dates, min_months) -> pd.DataFrame:
    """Original per-ticker loop implementation, kept as the test oracle."""
    out = pd.DataFrame(True, index=dates, columns=tickers)
    for tkr in tickers:
        ld = list_dates.get(tkr)
        if pd.isna(ld):
            continue
        out[tkr] = dates >= (ld + pd.DateOffset(months=min_months))
    return out


@pytest.fixture
def calendar() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", "2020-12-31", freq="BME")


def test_vectorized_matches_loop(calendar):
    """Vectorized mask must be identical to the original loop, NaT included."""
    tickers = pd.Index(["A", "B", "C", "D"])
    list_dates = pd.to_datetime(
        pd.Series({"A": "2019-06-01", "B": "2020-08-15", "C": pd.NaT, "D": "2020-01-01"})
    )
    vec = _listing_age_mask(calendar, tickers, list_dates, min_months=3)
    ref = _reference_loop(calendar, tickers, list_dates, min_months=3)
    assert vec.equals(ref)


def test_unknown_list_date_is_seasoned(calendar):
    """A ticker with an unknown (NaT) list date is treated as fully seasoned."""
    tickers = pd.Index(["C"])
    list_dates = pd.to_datetime(pd.Series({"C": pd.NaT}))
    mask = _listing_age_mask(calendar, tickers, list_dates, min_months=3)
    assert mask["C"].all()


def test_seasoning_threshold_boundary(calendar):
    """Stock becomes eligible exactly ``min_months`` after listing, not before."""
    tickers = pd.Index(["B"])
    list_dates = pd.to_datetime(pd.Series({"B": "2020-01-15"}))
    mask = _listing_age_mask(calendar, tickers, list_dates, min_months=3)
    # threshold = 2020-04-15 -> Jan/Feb/Mar False, Apr month-end (>= 4-15) True
    eligible_from = mask["B"][mask["B"]].index.min()
    assert eligible_from == pd.Timestamp("2020-04-30")
    assert not mask["B"].loc[:"2020-03-31"].any()


def test_shape_and_labels(calendar):
    """Output panel is aligned to the input calendar and ticker set."""
    tickers = pd.Index(["A", "B"])
    list_dates = pd.to_datetime(pd.Series({"A": "2019-01-01", "B": "2019-01-01"}))
    mask = _listing_age_mask(calendar, tickers, list_dates, min_months=3)
    assert mask.shape == (len(calendar), len(tickers))
    assert list(mask.columns) == list(tickers)
    assert mask.index.equals(calendar)
