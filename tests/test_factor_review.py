import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest import factor_review as fr

ZC = {"window": 500, "min_periods": 250, "clip": 3.0}
IDX = pd.bdate_range("2000-01-03", "2016-12-30")
H = 126


def _setup():
    """Factor A predicts dy, B is noise, C predicts with the wrong sign."""
    rng = np.random.default_rng(0)
    signal = pd.Series(rng.normal(size=len(IDX)), index=IDX).rolling(60).mean().bfill()
    z = pd.DataFrame({"A": signal + rng.normal(scale=0.3, size=len(IDX)),
                      "B": rng.normal(size=len(IDX)),
                      "C": -signal + rng.normal(scale=0.3, size=len(IDX))}, index=IDX)
    dy = signal * 50
    dy.iloc[-H:] = np.nan
    lib = pd.DataFrame({"name": ["a", "b", "c"], "theme": ["t1", "t1", "t2"], "sign": [1, 1, 1],
                        "weight_2y": [1.0, 1.0, 1.0]}, index=["A", "B", "C"])
    tw = pd.Series({"t1": 1.0, "t2": 1.0})
    return z, lib, tw, dy


def test_factor_table_flags_wrong_signed_factor():
    z, lib, tw, dy = _setup()
    t = fr.factor_table(z, lib, tw, dy, "2y", IDX[IDX >= "2010-01-01"], {"A", "B", "C"}, ZC)
    assert t.loc["A", "ic_2005"] > 0.2 and t.loc["C", "ic_2005"] < -0.2
    assert t.loc["C", "eras_negative"] == 3  # synthetic data ends 2016: the 2020+ era is empty
    assert t.loc["C", "loo_delta_ic"] > 0  # composite improves without the wrong-signed factor
    assert t.loc["A", "loo_delta_ic"] < 0
    assert t.loc["A", "closest"] == "C"


def test_theme_leave_one_out_signs():
    z, lib, tw, dy = _setup()
    d = fr.theme_leave_one_out(z, lib, tw, dy, "2y", {"A", "B", "C"}, ZC)
    assert d["t2"] > 0 and d["t1"] < 0


def test_nested_pruning_drops_wrong_signed_factor_and_is_causal():
    z, lib, tw, dy = _setup()
    rule = fr.PRUNING_RULES["drop_ic_below_0"]
    comp, dec = fr.nested_pruning(z, lib, tw, dy, "2y", {"A", "B", "C"}, ZC, H, rule)
    assert "C" in dec[2012]["dropped"] and "A" not in dec[2012]["dropped"]
    assert comp.loc[:"2011-12-31"].isna().all() and comp.loc["2012-01-03":].notna().any()
    # corrupting targets that were not yet realized at the 2014 decision must not change it
    start = pd.Timestamp("2014-01-01")
    cutoff = IDX[IDX.get_loc(IDX[IDX >= start][0]) - H]
    dy2 = dy.copy()
    dy2.loc[cutoff + pd.Timedelta(days=1):] = -dy2.loc[cutoff + pd.Timedelta(days=1):]
    _, dec2 = fr.nested_pruning(z, lib, tw, dy2, "2y", {"A", "B", "C"}, ZC, H, rule)
    assert dec2[2014] == dec[2014]


def test_flip_rule_flips_strongly_wrong_factor():
    z, lib, tw, dy = _setup()
    _, dec = fr.nested_pruning(z, lib, tw, dy, "2y", {"A", "B", "C"}, ZC, H, fr.PRUNING_RULES["drop_-0.10_flip_-0.20"])
    assert dec[2012]["flipped"] == ["C"]
