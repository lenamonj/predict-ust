import numpy as np
import pandas as pd

from ust_rv.backtest import selection as sel

IDX = pd.bdate_range("2009-01-01", "2016-12-31")


def _cands():
    """Two candidates: phase1 earns until 2011 then loses; ridge the reverse. Both qualify on constraints."""
    keys = [("phase1", 0.5, 0.2), ("ridge", 0.5, 0.2)]
    rv = pd.DataFrame({k: np.tile([1.0] * 100 + [0.0] * 100, len(IDX) // 200 + 1)[: len(IDX)] for k in keys}, index=IDX)
    rng = np.random.default_rng(0)
    early = IDX < "2011-01-01"
    a = np.where(early, 1.0, -1.0) + rng.normal(scale=0.5, size=len(IDX))
    b = np.where(early, -1.0, 1.0) + rng.normal(scale=0.5, size=len(IDX))
    pnl = pd.DataFrame({keys[0]: a, keys[1]: b}, index=IDX) * rv.to_numpy()
    rv.columns = pd.MultiIndex.from_tuples(keys)
    pnl.columns = pd.MultiIndex.from_tuples(keys)
    return sel.Candidates(rv, pnl)


def test_stats_use_only_history_before_selection_date():
    c = _cands()
    s1 = sel._stats(c, pd.Timestamp("2012-01-02"))
    c2 = sel.Candidates(c.rv.copy(), c.pnl.copy())
    c2.pnl.loc["2012-01-02":] = 1e6  # the future must not move the selection
    s2 = sel._stats(c2, pd.Timestamp("2012-01-02"))
    pd.testing.assert_frame_equal(s1, s2)


def test_select_prefers_qualifiers_then_sharpe():
    stats = pd.DataFrame({"sharpe": [2.0, 0.5, 0.8], "viol": [0.3, 0.0, 0.0]},
                         index=pd.MultiIndex.from_tuples([("phase1", 0.3, 0.1), ("phase1", 0.5, 0.2), ("ridge", 0.6, 0.2)]))
    assert sel.select(stats) == ("ridge", 0.6, 0.2)
    assert sel.select(stats, allowed=lambda k: k[0] == "phase1") == ("phase1", 0.5, 0.2)


def test_nested_switches_after_evidence_changes_and_is_out_of_sample():
    c = _cands()
    dy = {"2y": pd.Series(0.0, index=IDX)}
    r = sel.nested({"2y": c}, dy, (0.5, 0.2))
    picks = r.picks.loc[("c_per_tenor_model_pair", "2y")]
    assert picks[2012] == ("phase1", 0.5, 0.2)  # 2009-2011: two good years, one bad
    # expanding window: the switch comes only once cumulative evidence turns
    assert picks[2014] == ("ridge", 0.5, 0.2)
    assert picks[2016] == ("ridge", 0.5, 0.2)
    nested_pnl = r.pnl[("c_per_tenor_model_pair", "2y")]
    assert nested_pnl.index[0] == pd.Timestamp("2012-01-02")
    pd.testing.assert_series_equal(nested_pnl.loc["2012"], c.pnl.loc["2012", ("phase1", 0.5, 0.2)], check_names=False)
