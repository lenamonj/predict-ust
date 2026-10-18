"""Phase 1 acceptance checks against live data (Build_Plan Section 13)."""
import numpy as np
import pandas as pd
import pytest
import requests

from ust_rv.config import factors
from ust_rv.features.transforms import apply_transform
from ust_rv.panel import build_panel
from ust_rv.providers.alfred import URL, AlfredProvider, _api_key
from ust_rv.scoring.scorecard import compute_scores, scorecard

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def panel():
    return build_panel()


def _vintage_as_of(series_id: str, day: pd.Timestamp) -> pd.Series:
    """Independent path: the series exactly as FRED published it on `day`."""
    d = f"{day:%Y-%m-%d}"
    params = {"series_id": series_id, "api_key": _api_key(), "file_type": "json", "realtime_start": d, "realtime_end": d}
    obs = requests.get(URL, params=params, timeout=120).json()["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs})
    return pd.to_numeric(s, errors="coerce").dropna()


def test_payrolls_never_visible_before_release(panel):
    v = AlfredProvider().get_vintages("PAYEMS").vintages
    first_release = v.groupby("ref_date")["release_date"].min()
    # B1 reading on each grid day must come from a ref month already released that day
    as_of = panel.as_of["B1"].dropna()
    pit_days = panel.pit["B1"].astype("boolean").fillna(False).astype(bool)
    as_of = as_of[pit_days.reindex(as_of.index)]
    assert (as_of <= as_of.index).all()
    # spot check: the Sep-2026 print is invisible the day before its release and visible on release day
    ref = first_release.index.max()
    rel = first_release[ref]
    grid = panel.values.index
    before = grid[grid < rel][-1]
    on = grid[grid >= rel][0]
    assert panel.as_of.loc[before, "B1"] < rel
    assert panel.as_of.loc[on, "B1"] == rel


def test_lookahead_audit_50_pairs(panel):
    lib = {f["id"]: f for f in factors()}
    eligible = [
        fid for fid, f in lib.items()
        if f["source"]["primary"].get("provider") == "alfred" and "inputs" not in f["source"]["primary"]
        and panel.meta[fid]["source"] == "primary"
    ]
    rng = np.random.default_rng(20261018)
    pairs = []
    while len(pairs) < 50:
        fid = eligible[rng.integers(len(eligible))]
        pit = panel.pit[fid].astype("boolean").fillna(False).astype(bool)
        days = pit[pit].index
        days = days[days < panel.date]
        pairs.append((fid, days[rng.integers(len(days))]))

    mismatches = []
    for fid, day in pairs:
        f = lib[fid]
        spec = f["source"]["primary"].get("transform", f["transform"])
        expected = apply_transform(_vintage_as_of(f["source"]["primary"]["series"], day), spec).dropna().iloc[-1]
        got = panel.values.loc[day, fid]
        if not np.isclose(got, expected, rtol=1e-9, atol=1e-9):
            mismatches.append((fid, f"{day:%Y-%m-%d}", got, expected))
    assert not mismatches, mismatches


def test_scorecard_has_25_fresh_factors_per_tenor(panel):
    card = scorecard(panel, compute_scores(panel))
    for tenor in ("2y", "10y"):
        n = int(((card[f"w_{tenor}"] > 0) & card["fresh"] & card["z"].notna()).sum())
        assert n >= 25, (tenor, n)
    bbg_only = card.loc[["A6", "F6", "G6"], "status"]
    assert (bbg_only == "unavailable").all()
