import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest import normalization as nz

IDX = pd.bdate_range("2000-01-03", periods=4000)


def _values(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(IDX), 3)).cumsum(axis=0), index=IDX, columns=["A", "B", "C"])


@pytest.mark.parametrize("scheme", nz.ALL_SCHEMES, ids=lambda s: s.name)
def test_every_scheme_is_trailing_only(scheme):
    v = _values()
    full = nz.score(v, scheme)
    cut = 3000
    v2 = v.copy()
    v2.iloc[cut:] = -v2.iloc[cut:] * 5
    pd.testing.assert_frame_equal(full.iloc[:cut], nz.score(v2, scheme).iloc[:cut])


def test_percentile_is_on_z_scale():
    rng = np.random.default_rng(1)
    v = pd.DataFrame({"A": rng.normal(size=len(IDX))}, index=IDX)
    p = nz.percentile_score(v, 252)["A"].dropna()
    assert abs(p.mean()) < 0.05 and 0.9 < p.std() < 1.1
    assert p.abs().max() <= 3.0


def test_live_scheme_matches_rolling_z():
    from ust_rv.features.zscore import rolling_z

    v = _values()
    pd.testing.assert_frame_equal(nz.score(v, nz.LIVE), rolling_z(v, 2520, 1260, 3.0))


def test_nested_scheme_selection_uses_only_prior_pnl():
    idx = pd.bdate_range("2009-01-01", "2016-12-30")
    rng = np.random.default_rng(2)
    early = idx < "2012-01-01"
    pnl = pd.DataFrame({"a": np.where(early, 1.0, -1.0) + rng.normal(scale=0.5, size=len(idx)),
                        "b": np.where(early, -1.0, 1.0) + rng.normal(scale=0.5, size=len(idx))}, index=idx)
    rv = pd.DataFrame(1.0, index=idx, columns=["a", "b"])
    _, _, picks = nz.nested_scheme_selection(pnl, rv, ["a", "b"])
    assert picks[2012] == "a"  # only 2009-2011 known: 'a' looked best
    pnl2 = pnl.copy()
    pnl2.loc["2013-01-01":] = 0.0
    _, _, picks2 = nz.nested_scheme_selection(pnl2, rv, ["a", "b"])
    assert picks2[2012] == picks[2012] and picks2[2013] == picks[2013]
