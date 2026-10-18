import os

import pandas as pd
import pytest

from ust_rv.providers import public
from ust_rv.providers.alfred import AlfredProvider
from ust_rv.providers.base import ParquetCache, ProviderError, cached_fetch
from ust_rv.providers.bloomberg import BloombergProvider
from ust_rv.providers.fred import FredProvider


@pytest.fixture
def cache(tmp_path):
    return ParquetCache(tmp_path)


def _frame(n=3):
    d = pd.date_range("2024-01-01", periods=n)
    return pd.DataFrame({"ref_date": d, "release_date": d, "value": range(n)})


# ---------- cache behaviour ----------

def test_cache_round_trip(cache):
    cache.write("p", "S/1", _frame())
    pd.testing.assert_frame_equal(cache.read("p", "S/1"), _frame())
    assert cache.fetched_today("p", "S/1")


def test_cached_fetch_uses_cache_same_day(cache):
    cache.write("p", "k", _frame())
    calls = []
    df, stale, err = cached_fetch(cache, "p", "k", lambda c: calls.append(1) or _frame(5))
    assert calls == [] and len(df) == 3 and not stale


def test_cached_fetch_failure_returns_stale_cache(cache):
    cache.write("p", "k", _frame())
    os.utime(cache.path("p", "k"), (86400, 86400))  # mtime in 1970: not fetched today

    def boom(_):
        raise ConnectionError("down")

    df, stale, err = cached_fetch(cache, "p", "k", boom)
    assert stale and "down" in err and len(df) == 3


def test_cached_fetch_failure_without_cache_raises(cache):
    def boom(_):
        raise ConnectionError("down")

    with pytest.raises(ProviderError):
        cached_fetch(cache, "p", "missing", boom)


def test_bloomberg_without_terminal_or_cache_raises(cache):
    with pytest.raises(ProviderError):
        BloombergProvider(cache).get_vintages("LF98OAS Index")


# ---------- FRED incremental and ALFRED parsing with fake clients ----------

class FakeFred:
    def __init__(self):
        self.starts = []

    def get_series(self, series_id, observation_start=None):
        self.starts.append(observation_start)
        idx = pd.date_range("2024-01-01", "2024-03-31")
        if observation_start is not None:
            idx = idx[idx >= observation_start]
        return pd.Series(1.0, index=idx)



def test_fred_incremental_refetch_from_last_date_minus_30(cache):
    fake = FakeFred()
    p = FredProvider(cache, client=fake)
    first = p._fetch("X", None)
    second = p._fetch("X", first)
    assert fake.starts[1] == pd.Timestamp("2024-03-31") - pd.Timedelta(days=30)
    pd.testing.assert_frame_equal(first, second)


ALFRED_ROWS = [
    {"realtime_start": "2024-02-02", "realtime_end": "2024-03-07", "date": "2024-01-01", "value": "100"},
    {"realtime_start": "2024-03-08", "realtime_end": "9999-12-31", "date": "2024-01-01", "value": "101"},
    {"realtime_start": "2024-03-08", "realtime_end": "9999-12-31", "date": "2024-02-01", "value": "."},
]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


def _paged_get(rows, page_size, reported=None):
    calls = []

    def get(url, params, timeout):
        calls.append(params["offset"])
        chunk = rows[params["offset"]: params["offset"] + page_size]
        return FakeResponse({"count": reported if reported is not None else len(rows), "observations": chunk})

    return get, calls


def test_alfred_paginates_and_parses(cache, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "x")
    get, calls = _paged_get(ALFRED_ROWS, page_size=2)
    df = AlfredProvider(cache, get=get).get_vintages("X").vintages
    assert calls == [0, 2]
    assert list(df.columns) == ["ref_date", "release_date", "value"]
    assert df["value"].isna().sum() == 1  # "." becomes NaN
    assert df.loc[1, "release_date"] == pd.Timestamp("2024-03-08")


def test_alfred_row_count_mismatch_is_hard_error(cache, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "x")
    get, _ = _paged_get(ALFRED_ROWS, page_size=10, reported=5)
    with pytest.raises(ProviderError, match="API reported 5"):
        AlfredProvider(cache, get=get).get_vintages("X")


# ---------- public feed parsers ----------

def _td_record(day, term, btc, ind, comp, amount, tips="No", frn="No"):
    return {
        "competitiveAuctionCloseDate": f"{day}T00:00:00",
        "securityType": "Note",
        "originalSecurityTerm": term,
        "offeringAmount": str(amount),
        "bidToCoverRatio": btc,
        "indirectBiddersCompetitiveParAwarded": ind,
        "subtotalCompetitiveParAwardedInsideOfOfferingAmount": comp,
        "inflationProtected": tips,
        "floatingRate": frn,
    }


TD = public.parse_treasurydirect(
    [
        _td_record("2024-01-10", "10-Year", "2.5", "60", "100", 40e9),
        _td_record("2024-01-25", "2-Year", "2.7", "", "", 60e9),
        _td_record("2024-02-08", "10-Year", "2.4", "70", "100", 42e9),
        _td_record("2024-02-15", "10-Year", "2.1", "", "", 20e9, tips="Yes"),
        _td_record("2024-03-05", "2-Year", "2.6", "", "", 61e9),
    ]
)


def test_auction_series_bid_to_cover_and_indirect():
    btc = public.auction_series(TD, "10-Year", "bid_to_cover")
    assert list(btc["value"]) == [2.5, 2.4]  # TIPS excluded
    assert (btc["release_date"] == btc["ref_date"]).all()
    ind = public.auction_series(TD, "10-Year", "indirect_share")
    assert list(ind["value"]) == [60.0, 70.0]
    assert public.auction_series(TD, "2-Year", "indirect_share").empty


def test_coupon_issuance_monthly_drops_current_month():
    out = public.coupon_issuance(TD, pd.Timestamp("2024-03-20"))
    assert list(out["value"]) == [100.0, 42.0]  # Jan 40+60, Feb 42 (TIPS excluded), Mar dropped
    assert out.loc[0, "release_date"] == pd.Timestamp("2024-01-31")


def test_cftc_net_pct_oi_and_friday_release():
    recs = [
        {"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "cftc_contract_market_code": "043602",
         "lev_money_positions_long": "100", "lev_money_positions_short": "300", "open_interest_all": "1000"},
        {"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "cftc_contract_market_code": "043607",
         "lev_money_positions_long": "50", "lev_money_positions_short": "50", "open_interest_all": "1000"},
        {"report_date_as_yyyy_mm_dd": "2024-01-02T00:00:00.000", "cftc_contract_market_code": "042601",
         "lev_money_positions_long": "10", "lev_money_positions_short": "0", "open_interest_all": "100"},
    ]
    pos = public.parse_cftc(recs)
    ty = public.cftc_levnet_pct_oi(pos, ["TY", "TN"])
    assert ty.loc[0, "value"] == pytest.approx(-200 / 2000 * 100)
    assert ty.loc[0, "release_date"] == pd.Timestamp("2024-01-05")  # Friday
    assert public.cftc_levnet_pct_oi(pos, ["TU"]).loc[0, "value"] == pytest.approx(10.0)


def test_nyfed_release_lag():
    out = public.parse_nyfed({"pd": {"timeseries": [{"asofdate": "2024-01-03", "keyid": "K", "value": "-123"}]}})
    assert out.loc[0, "release_date"] == pd.Timestamp("2024-01-11")
    assert out.loc[0, "value"] == -123.0


# ---------- live pulls ----------

@pytest.mark.network
def test_live_fred(cache):
    df = FredProvider(cache).get_vintages("DGS10").vintages
    assert df["ref_date"].min() <= pd.Timestamp("1962-01-03")
    assert df["ref_date"].max() >= pd.Timestamp.today() - pd.Timedelta(days=10)


@pytest.mark.network
def test_live_alfred(cache):
    df = AlfredProvider(cache).get_vintages("PAYEMS").vintages
    assert df["release_date"].nunique() > 500
    assert not df["ref_date"].is_unique


@pytest.mark.network
@pytest.mark.parametrize("sid,start_before", [
    ("TD:BTC_10Y", "2001-01-01"),
    ("TD:INDIRECT_10Y", "2009-01-01"),
    ("TD:BTC_2Y", "2001-01-01"),
    ("TD:COUPON_ISSUANCE", "1990-01-01"),
    ("CFTC:LEVNET_TU", "2006-12-31"),
    ("CFTC:LEVNET_TY_TN", "2006-12-31"),
    ("NYFED:PDPOSGSC-G7L11", "2013-12-31"),
])
def test_live_public(cache, sid, start_before):
    df = public.PublicProvider(cache).get_vintages(sid).vintages
    assert df["ref_date"].min() <= pd.Timestamp(start_before)
    assert df["release_date"].max() >= pd.Timestamp.today() - pd.Timedelta(days=60)
    assert df["value"].notna().all()
