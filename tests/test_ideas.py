import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest import ideas

ZC = {"window": 500, "min_periods": 250, "clip": 3.0}
IDX = pd.bdate_range("2000-01-03", periods=2000)


def test_slope_values_in_bp():
    y = pd.DataFrame({"y2": [4.0, 4.5], "y10": [4.25, 4.0]})
    assert ideas.slope_values(y).tolist() == pytest.approx([25.0, -50.0])


def test_with_slope_adds_f7_for_10y_only_with_negative_sign():
    lib = pd.DataFrame({"name": ["a"], "theme": ["valuation"], "sign": [1], "weight_2y": [1.0], "weight_10y": [1.0]}, index=["A"])
    values = pd.DataFrame({"A": [1.0, 2.0]})
    y = pd.DataFrame({"y2": [4.0, 4.0], "y10": [5.0, 4.5]})
    v, lib2 = ideas.with_slope(values, lib, y)
    assert v["F7"].tolist() == pytest.approx([100.0, 50.0])
    assert lib2.loc["F7", "sign"] == -1 and lib2.loc["F7", "weight_2y"] == 0 and lib2.loc["F7", "weight_10y"] == 1.0


def test_blend_is_trailing_and_equal_weight():
    rng = np.random.default_rng(0)
    comp = pd.Series(rng.normal(size=len(IDX)), index=IDX)
    mom = pd.Series(rng.normal(size=len(IDX)).cumsum(), index=IDX)
    b = ideas.blend(comp, mom, ZC)
    comp2, mom2 = comp.copy(), mom.copy()
    comp2.iloc[1500:] = 9.0
    mom2.iloc[1500:] = -9.0
    pd.testing.assert_series_equal(b.iloc[:1500], ideas.blend(comp2, mom2, ZC).iloc[:1500])
    # equal inputs give back the (re-standardized) input
    same = ideas.blend(comp, comp, ZC)
    assert same.dropna().corr(comp.reindex(same.dropna().index)) > 0.99
