import zlib

import numpy as np
import pandas as pd

from ust_rv.providers.base import Fetch, ProviderError
from ust_rv.runner import run

END1, END2 = "2020-06-01", "2020-06-02"


class SyntheticFred:
    """Random-walk daily series long enough for three stacked 5y z-score windows."""

    def __init__(self, shift: float = 0.0):
        self.shift = shift

    def get_vintages(self, series_id):
        d = pd.bdate_range("1996-01-01", "2020-06-30")  # factor, theme and composite z each need 5y
        rng = np.random.default_rng(zlib.crc32(series_id.encode()))
        v = 3.0 + np.cumsum(rng.normal(scale=0.03, size=len(d)))
        v[-25:] += self.shift
        return Fetch(pd.DataFrame({"ref_date": d, "release_date": d, "value": v}))


class Missing:
    def get_vintages(self, series_id):
        raise ProviderError("no terminal")


LIB = [
    {"id": "X1", "name": "x1", "theme": "growth", "source": {"primary": {"provider": "fred", "series": "S1"}, "fallback": None},
     "transform": {"type": "diff", "periods": 21}, "sign": 1, "weight_2y": 1.0, "weight_10y": 1.0, "max_age_days": 5},
    {"id": "X2", "name": "x2", "theme": "inflation", "source": {"primary": {"provider": "fred", "series": "S2"}, "fallback": None},
     "transform": {"type": "level"}, "sign": -1, "weight_2y": 1.0, "weight_10y": 0.5, "max_age_days": 5},
    {"id": "A6", "name": "surprise", "theme": "growth", "source": {"primary": {"provider": "bloomberg", "series": "CESIUSD Index"}, "fallback": None},
     "transform": {"type": "level"}, "sign": 1, "weight_2y": 1.0, "weight_10y": 1.0, "max_age_days": 5},
]


def _providers(shift=0.0):
    return {"fred": SyntheticFred(shift), "alfred": SyntheticFred(shift), "bloomberg": Missing(), "public": Missing()}


def test_run_persists_and_diffs(tmp_path):
    r1 = run(_providers(), LIB, runs_dir=tmp_path, today=END1)
    assert (tmp_path / "snapshots" / "2020-06-01.parquet").exists()
    assert r1.prior is None and r1.movers.empty
    assert r1.signal["2y"]["rv"] in (-1, 0, 1)
    assert r1.signal["curve"] in ("steepener", "flattener", "neutral")
    assert r1.unverified_tickers == ["CESIUSD Index"]
    assert r1.provider_log.loc["A6", "status"] == "unavailable"

    r2 = run(_providers(shift=1.0), LIB, runs_dir=tmp_path, today=END2)
    assert r2.prior["run_date"] == pd.Timestamp(END1)
    assert r2.prior["rv_2y"] == r1.signal["2y"]["rv"]
    assert "X2" in r2.movers.index  # level shift of +1 moves the z of X2

    hist = pd.read_parquet(tmp_path / "signal_history.parquet")
    assert len(hist) == 2


def test_same_day_rerun_overwrites_snapshot(tmp_path):
    run(_providers(), LIB, runs_dir=tmp_path, today=END1)
    run(_providers(shift=1.0), LIB, runs_dir=tmp_path, today=END1)
    snaps = list((tmp_path / "snapshots").glob("*.parquet"))
    assert len(snaps) == 1
    assert len(pd.read_parquet(tmp_path / "signal_history.parquet")) == 2
