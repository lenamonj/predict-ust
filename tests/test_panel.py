import pandas as pd
import pytest

from ust_rv.config import factors, settings
from ust_rv.features.transforms import TRANSFORMS
from ust_rv.panel import build_panel
from ust_rv.providers.base import Fetch, ProviderError
from ust_rv.sources import business_grid

PROVIDERS = {"fred", "alfred", "bloomberg", "public", "model"}


# ---------- factors.yaml schema ----------

def _inputs(src):
    return src.get("inputs", [src])


def test_factor_library_schema():
    lib = factors()
    themes = set(settings()["theme_weights"])
    assert len(lib) == 46
    assert len({f["id"] for f in lib}) == 46
    for f in lib:
        assert f["theme"] in themes, f["id"]
        assert f["sign"] in (1, -1), f["id"]
        assert f["weight_2y"] >= 0 and f["weight_10y"] >= 0, f["id"]
        assert f["max_age_days"] > 0, f["id"]
        assert len(f.get("definition", "")) > 80, f"{f['id']} needs a full definition"
        assert f["transform"]["type"] in TRANSFORMS, f["id"]
        for which in ("primary", "fallback"):
            src = f["source"][which]
            if src is None:
                continue
            if "inputs" in src:
                assert src["combine"] in ("diff", "ratio") and len(src["inputs"]) == 2, f["id"]
            for inp in _inputs(src):
                assert inp["provider"] in PROVIDERS, f["id"]
                if "transform" in inp:
                    assert inp["transform"]["type"] in TRANSFORMS, f["id"]


def test_phase1_factor_counts_without_bloomberg():
    """Factors usable with no terminal (some non-Bloomberg source) and non-zero weight."""
    def usable(f):
        return any(
            f["source"][w] is not None and all(i["provider"] != "bloomberg" for i in _inputs(f["source"][w]))
            for w in ("primary", "fallback")
        )
    lib = factors()
    assert sum(usable(f) and f["weight_2y"] > 0 for f in lib) >= 25
    assert sum(usable(f) and f["weight_10y"] > 0 for f in lib) >= 25


# ---------- panel with fake providers ----------

class FakeProvider:
    def __init__(self, tables: dict[str, pd.DataFrame], fail: set[str] = frozenset(), stale: set[str] = frozenset()):
        self.tables, self.fail, self.stale = tables, fail, stale

    def get_vintages(self, series_id):
        if series_id in self.fail or series_id not in self.tables:
            raise ProviderError(f"no {series_id}")
        return Fetch(self.tables[series_id], stale=series_id in self.stale)


def _daily(start="2020-01-01", end="2020-03-31", value=1.0):
    d = pd.bdate_range(start, end)
    return pd.DataFrame({"ref_date": d, "release_date": d, "value": value})


def _monthly_release_lagged():
    refs = pd.date_range("2019-11-01", "2020-02-01", freq="MS")
    return pd.DataFrame({"ref_date": refs, "release_date": refs + pd.Timedelta(days=40), "value": [1.0, 2.0, 3.0, 4.0]})


def _factor(fid, primary, fallback=None):
    return {"id": fid, "name": fid, "theme": "growth", "source": {"primary": primary, "fallback": fallback},
            "transform": {"type": "level"}, "sign": 1, "weight_2y": 1.0, "weight_10y": 1.0, "max_age_days": 10}


@pytest.fixture
def providers():
    fred = FakeProvider({"DGS1": _daily(value=1.8), "DGS2": _daily(value=2.0), "DGS7": _daily(value=2.7), "DGS10": _daily(value=3.0), "DFF": _daily(value=1.5), "NASDAQCOM": _daily(value=100.0), "BAA10Y": _daily(value=2.0),
                          "DGS6MO": _daily(value=1.6), "DGS3": _daily(value=2.2), "DGS20": _daily(value=3.2), "BAD": _daily()})
    alfred = FakeProvider({"M": _monthly_release_lagged()})
    bbg = FakeProvider({})
    return {"fred": fred, "alfred": alfred, "bloomberg": bbg, "public": FakeProvider({})}


def test_panel_status_fallback_and_combine(providers):
    lib = [
        _factor("X1", {"provider": "alfred", "series": "M"}),
        _factor("X2", {"provider": "bloomberg", "series": "T Index"}, {"combine": "diff", "inputs": [
            {"provider": "fred", "series": "DGS2"}, {"provider": "fred", "series": "DFF"}]}),
        _factor("X3", {"provider": "bloomberg", "series": "T Index"}),
        _factor("X4", {"provider": "fred", "series": "MISSING"}),
    ]
    p = build_panel(providers, lib, end="2020-03-31")
    assert p.meta["X1"]["status"] == "ok"
    assert p.meta["X2"]["source"] == "fallback" and p.meta["X2"]["label"] == "fred:DGS2 - fred:DFF"
    assert p.values["X2"].dropna().eq(0.5).all()
    assert p.meta["X3"]["status"] == "unavailable"
    assert p.meta["X4"]["status"] == "error" and p.meta["X4"]["errors"]
    assert p.values["X3"].isna().all()


def test_panel_monthly_value_appears_on_release_date(providers):
    p = build_panel(providers, [_factor("X1", {"provider": "alfred", "series": "M"})], end="2020-03-31")
    s = p.values["X1"]
    # ref 2020-01-01 released 2020-02-10 (Monday)
    assert s.loc["2020-02-07"] == 2.0
    assert s.loc["2020-02-10"] == 3.0
    assert p.as_of["X1"].loc["2020-02-10"] == pd.Timestamp("2020-02-10")


def test_panel_stale_flag(providers):
    providers["fred"].stale = {"DGS2"}
    p = build_panel(providers, [_factor("X5", {"provider": "fred", "series": "DGS2"})], end="2020-03-31")
    assert p.meta["X5"]["status"] == "stale"


def test_business_grid_skips_federal_holidays():
    g = business_grid("2024-01-01", "2024-01-31")
    assert pd.Timestamp("2024-01-01") not in g  # New Year
    assert pd.Timestamp("2024-01-15") not in g  # MLK
    assert pd.Timestamp("2024-01-02") in g


def test_pit_from_flags_backfilled_history(providers):
    f = _factor("X6", {"provider": "fred", "series": "DGS2", "pit_from": "2020-02-15"})
    p = build_panel(providers, [f], end="2020-03-31")
    pit = p.pit["X6"].dropna().astype(bool)
    assert not pit.loc[:"2020-02-14"].any()
    assert pit.loc["2020-02-18":].all()


def test_undatable_vintage_table_is_a_factor_error_not_a_crash(providers):
    # every ref in the first (and only revised) vintage: the engine cannot estimate a release lag
    refs = pd.date_range("2019-01-01", periods=3, freq="MS")
    vt = pd.DataFrame({"ref_date": list(refs) + [refs[0]], "release_date": [pd.Timestamp("2019-06-01")] * 3 + [pd.Timestamp("2019-07-01")],
                       "value": [1.0, 2.0, 3.0, 1.5]})
    providers["alfred"].tables["ONEVINTAGE"] = vt
    p = build_panel(providers, [_factor("X7", {"provider": "alfred", "series": "ONEVINTAGE"})], end="2020-03-31")
    assert p.meta["X7"]["status"] == "error"
    assert "release lag" in " ".join(p.meta["X7"]["errors"])


def test_market_date_is_last_actual_observation(providers):
    providers["fred"].tables["DGS10"] = _daily(end="2020-03-27", value=3.0)  # Friday; grid runs to Tuesday 31st
    p = build_panel(providers, [_factor("X8", {"provider": "fred", "series": "DGS2"})], end="2020-03-31")
    assert p.date == pd.Timestamp("2020-03-31")
    assert p.market_date == pd.Timestamp("2020-03-27")
