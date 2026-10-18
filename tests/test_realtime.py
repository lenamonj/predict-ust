import numpy as np
import pandas as pd
import pytest

from ust_rv.realtime import _vintage_readings, median_release_lag, realtime_readings, to_grid

LEVEL = {"type": "level"}
MCHG = {"type": "mean_change", "periods": 1}


def _payrolls_vintages() -> pd.DataFrame:
    """Monthly payrolls: ref M published ~5 weeks after month start, prior two months revised each print."""
    refs = pd.date_range("2020-01-01", periods=8, freq="MS")
    rows = []
    for k, ref in enumerate(refs):
        release = ref + pd.DateOffset(months=1) + pd.Timedelta(days=4)
        rows.append((ref, release, 1000.0 + 10 * k))
        for back in (1, 2):
            if k - back >= 0:
                rows.append((refs[k - back], release, 1000.0 + 10 * (k - back) + back))
    return pd.DataFrame(rows, columns=["ref_date", "release_date", "value"])


def test_ref_month_never_visible_before_release():
    vt = _payrolls_vintages()
    first_release = vt.groupby("ref_date")["release_date"].min()
    grid = pd.bdate_range("2020-01-01", "2020-12-31")
    g = to_grid(realtime_readings(vt, LEVEL), grid).dropna(subset=["value"])
    released_on = first_release.loc[g["ref_date"]].to_numpy()
    assert (released_on <= g.index.to_numpy()).all()
    assert (g["as_of"] <= g.index).all()


def test_reading_uses_vintage_values():
    vt = _payrolls_vintages()
    r = realtime_readings(vt, MCHG)
    # release of ref 2020-04: new value 1030, ref 2020-03 revised to 1021 in the same vintage
    rel = pd.Timestamp("2020-05-05")
    assert r.loc[rel, "value"] == pytest.approx(1030 - 1021)


def test_revision_changes_reading_only_from_revision_date():
    vt = _payrolls_vintages()
    last_ref = vt["ref_date"].max()
    bench = pd.Timestamp("2020-10-20")  # benchmark revision, no new ref month
    vt = pd.concat([vt, pd.DataFrame([(last_ref, bench, 5000.0)], columns=vt.columns)], ignore_index=True)
    grid = pd.bdate_range("2020-09-01", "2020-12-31")
    g = to_grid(realtime_readings(vt, LEVEL), grid)
    assert (g.loc[: bench - pd.Timedelta(days=1), "value"] != 5000.0).all()
    assert (g.loc[bench:, "value"] == 5000.0).all()


def test_pre_vintage_approximation():
    refs = pd.date_range("2000-01-01", "2001-12-01", freq="MS")
    first_vintage = pd.Timestamp("2001-03-15")
    lag = pd.Timedelta(days=35)
    rows = [(r, first_vintage, float(i)) for i, r in enumerate(refs) if r + lag <= first_vintage]
    rows += [(r, r + lag, float(i)) for i, r in enumerate(refs) if r + lag > first_vintage]
    rows += [(refs[0], pd.Timestamp("2001-06-01"), -1.0)]  # a revision makes the table revised
    vt = pd.DataFrame(rows, columns=["ref_date", "release_date", "value"])
    assert median_release_lag(vt) == lag
    r = realtime_readings(vt, LEVEL)
    approx = r[~r["pit"]]
    assert len(approx) > 0
    assert (approx.index == approx["ref_date"] + lag).all()
    assert (approx.index < first_vintage).all()
    assert r[r["pit"]].index.min() == first_vintage


def test_unrevised_backfilled_series_gets_pseudo_release_dates():
    """ALFRED series never revised after a history backfill must still be approximated pre-vintage."""
    refs = pd.date_range("2000-01-01", "2001-12-01", freq="MS")
    first_vintage = pd.Timestamp("2001-03-15")
    lag = pd.Timedelta(days=35)
    rows = [(r, first_vintage if r + lag <= first_vintage else r + lag, float(i)) for i, r in enumerate(refs)]
    vt = pd.DataFrame(rows, columns=["ref_date", "release_date", "value"])
    r = realtime_readings(vt, LEVEL)
    approx = r[~r["pit"]]
    assert len(approx) == sum(ref + lag < first_vintage for ref in refs)
    assert (approx.index == approx["ref_date"] + lag).all()
    assert r.index.is_monotonic_increasing


def test_approx_lag_override():
    vt = pd.DataFrame(
        [
            (pd.Timestamp("2000-01-01"), pd.Timestamp("2000-06-01"), 1.0),
            (pd.Timestamp("2000-04-01"), pd.Timestamp("2000-06-01"), 2.0),
            (pd.Timestamp("2000-07-01"), pd.Timestamp("2000-08-01"), 3.0),
            (pd.Timestamp("2000-07-01"), pd.Timestamp("2000-09-01"), 3.5),
        ],
        columns=["ref_date", "release_date", "value"],
    )
    r = realtime_readings(vt, LEVEL, approx_lag_days=120)
    approx = r[~r["pit"]]
    assert list(approx.index) == [pd.Timestamp("2000-01-01") + pd.Timedelta(days=120)]


def test_fast_path_matches_vintage_path_for_unrevised_data():
    idx = pd.bdate_range("2020-01-01", periods=300)
    vals = np.cumsum(np.random.default_rng(0).normal(size=300)) + 100
    vt = pd.DataFrame({"ref_date": idx, "release_date": idx, "value": vals})
    spec = {"type": "diff", "periods": 21}
    fast = realtime_readings(vt, spec)
    slow = _vintage_readings(vt, spec, approx_lag_days=0)
    pd.testing.assert_series_equal(fast["value"], slow["value"], check_names=False)


def test_weekend_release_lands_next_business_day():
    r = pd.DataFrame(
        {"value": [1.0], "ref_date": [pd.Timestamp("2024-01-01")], "pit": [True]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-06")], name="release_date"),  # Saturday
    )
    g = to_grid(r, pd.bdate_range("2024-01-04", "2024-01-09"))
    assert g["value"].isna().sum() == 2
    assert g.loc["2024-01-08", "value"] == 1.0
