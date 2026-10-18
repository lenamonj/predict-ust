import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest import baselines, engine

IDX = pd.bdate_range("2008-01-01", "2016-12-31")


def test_baselines():
    assert (baselines.always_long(IDX[:3]) == 1).all()
    assert (baselines.random_walk(IDX[:3]) == 0).all()
    mom = baselines.momentum(pd.Series([25.0, -10.0, 0.0], index=IDX[:3]))
    assert mom.tolist() == [-1.0, 1.0, 0.0]  # rising yields -> short duration


def test_sweep_has_full_grid_and_section10_columns():
    rng = np.random.default_rng(0)
    comp = pd.Series(rng.normal(size=len(IDX)), index=IDX).rolling(60).mean() * 3
    dy = comp * 10 + rng.normal(size=len(IDX))
    s = engine.sweep(comp, dy, lambda rv: pd.Series(0.1, index=rv.index) * rv)
    assert len(s) == 20  # every (enter, exit) pair with exit <= enter
    assert {"enter", "exit", "neutral_share", "avg_hold_days", "hit_rate", "ic", "overlay_sharpe"} <= set(s.columns)
    tight = s[(s["enter"] == 0.3) & (s["exit"] == 0.1)].iloc[0]
    wide = s[(s["enter"] == 0.75) & (s["exit"] == 0.3)].iloc[0]
    assert tight["neutral_share"] < wide["neutral_share"]


def _sweep_table(rows):
    return pd.DataFrame(rows, columns=["enter", "exit", "neutral_share", "avg_hold_days", "overlay_sharpe"])


def test_select_thresholds_prefers_qualifying_pair_with_best_sharpe():
    a = _sweep_table([(0.3, 0.1, 0.20, 100, 2.0), (0.5, 0.25, 0.45, 80, 0.5), (0.6, 0.25, 0.50, 90, 0.8)])
    b = _sweep_table([(0.3, 0.1, 0.25, 100, 2.0), (0.5, 0.25, 0.42, 70, 0.6), (0.6, 0.25, 0.52, 95, 0.7)])
    enter, exit_, ok = engine.select_thresholds({"2y": a, "10y": b})
    assert (enter, exit_, ok) == (0.6, 0.25, True)  # (0.3, 0.1) has the best Sharpe but fails neutral share


def test_select_thresholds_falls_back_to_least_violation():
    a = _sweep_table([(0.3, 0.1, 0.10, 100, 2.0), (0.5, 0.25, 0.35, 80, 0.5)])
    b = _sweep_table([(0.3, 0.1, 0.10, 100, 2.0), (0.5, 0.25, 0.38, 80, 0.6)])
    enter, exit_, ok = engine.select_thresholds({"2y": a, "10y": b})
    assert (enter, exit_, ok) == (0.5, 0.25, False)


def test_windows():
    pit = pd.Series(np.where(IDX >= "2014-01-01", 0.95, 0.7), index=IDX)
    w = engine.windows(pit)
    assert w["2009+"][0] >= pd.Timestamp("2009-01-01")
    assert w["PIT>=90%"][0] >= pd.Timestamp("2014-01-01")
    assert w["2005+"][0] == IDX[0]
