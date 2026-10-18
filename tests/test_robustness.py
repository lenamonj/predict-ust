import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest import metrics
from ust_rv.backtest import robustness as rb
from ust_rv.scoring.composite import weighted_average

ZC = {"window": 500, "min_periods": 250, "clip": 3.0}
IDX = pd.bdate_range("2000-01-03", periods=2500)


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    z = pd.DataFrame(rng.normal(size=(len(IDX), 4)), index=IDX, columns=["A", "B", "C", "D"]).rolling(30).mean()
    weights = pd.Series({"A": 1.0, "B": 0.5, "C": 1.0, "D": 0.0})
    themes = pd.Series({"A": "t1", "B": "t1", "C": "t2", "D": "t2"})
    tw = pd.Series({"t1": 1.0, "t2": 2.0})
    return z, weights, themes, tw


def test_pnl_components_sum_to_overlay_pnl():
    idx = pd.bdate_range("2020-01-01", periods=30)
    rng = np.random.default_rng(1)
    y = pd.Series(4 + rng.normal(scale=0.02, size=30).cumsum(), index=idx)
    rv = pd.Series(np.sign(rng.normal(size=30)), index=idx)
    comp = metrics.pnl_components(rv, y, y - 0.3, 1.0, 2.0, y - 0.5)
    pd.testing.assert_series_equal(comp.sum(axis=1), metrics.overlay_pnl(rv, y, y - 0.3, 1.0, 2.0, y - 0.5))
    assert list(comp.columns) == ["direction", "carry", "roll"]


def test_drawdown_is_non_positive_and_zero_at_peaks():
    dd = metrics.drawdown(pd.Series([1.0, 2.0, -5.0, 1.0, 10.0]))
    assert dd.tolist() == [0.0, 0.0, -5.0, -4.0, 0.0]


def test_composite_decomposition_is_exact():
    z, weights, themes, tw = _setup()
    from ust_rv.features.zscore import rolling_z
    from ust_rv.scoring.composite import theme_pressure

    pressure = z
    tp = theme_pressure(pressure, weights, themes)
    tz = rolling_z(tp, ZC["window"], ZC["min_periods"], ZC["clip"])
    raw = weighted_average(tz, tw)
    comp = rolling_z(raw, ZC["window"], ZC["min_periods"], ZC["clip"])
    # scorecard-style contributions for the last date: w * pressure / sum |w| within theme
    last = pressure.iloc[-1]
    card = pd.DataFrame({"factor": list("ABCD"), "theme": themes, "w_2y": weights, "sign_2y": "PASS"})
    den = card["w_2y"].groupby(card["theme"]).transform(lambda w: w[w > 0].sum())
    card["contrib_2y"] = (weights * last / den).where(weights > 0)
    terms, k = rb.composite_decomposition(card, tp, tw, raw, "2y", ZC)
    total = terms["contribution"].sum() + k["theme_means"] + k["composite_mean"]
    assert total == pytest.approx(float(comp.iloc[-1]), abs=1e-9)
    assert k["residual"] == pytest.approx(0.0, abs=1e-9)


def test_placebo_is_reproducible_and_centered():
    z, weights, themes, tw = _setup()
    dy = pd.Series(np.random.default_rng(3).normal(size=len(IDX)), index=IDX)
    a = rb.placebo_ic(z, weights, themes, tw, dy, ZC, "2002-01-01", n=20, seed=7)
    b = rb.placebo_ic(z, weights, themes, tw, dy, ZC, "2002-01-01", n=20, seed=7)
    np.testing.assert_array_equal(a, b)
    assert abs(np.nanmean(a)) < 0.1  # unrelated target: no systematic IC


def test_nonoverlapping_ic_offsets_and_sizes():
    comp = pd.Series(np.arange(1260, dtype=float), index=pd.bdate_range("2010-01-01", periods=1260))
    no = rb.nonoverlapping_ic(comp, comp * 2, horizon=126, start="2010-01-01")
    assert len(no) == 126 and (no["n"] == 10).all()
    assert np.allclose(no["ic"], 1.0)


def test_factor_terms_at_matches_decomposition_today_and_is_causal():
    z, weights, themes, tw = _setup()
    from ust_rv.features.zscore import rolling_z
    from ust_rv.scoring.composite import theme_pressure

    tp = theme_pressure(z, weights, themes)
    raw = weighted_average(rolling_z(tp, ZC["window"], ZC["min_periods"], ZC["clip"]), tw)
    comp = rolling_z(raw, ZC["window"], ZC["min_periods"], ZC["clip"])
    for date in (IDX[-1], IDX[-127]):
        terms, k = rb.factor_terms_at(z, weights, themes, tp, tw, raw, date, ZC)
        assert terms.sum() + k["theme_means"] + k["composite_mean"] == pytest.approx(float(comp.loc[date]), abs=1e-9)
        assert k["composite_z"] == pytest.approx(float(comp.loc[date]), abs=1e-9)
    # past decomposition ignores later data
    z2 = z.copy()
    z2.iloc[-100:] = 50.0
    tp2 = theme_pressure(z2, weights, themes)
    raw2 = weighted_average(rolling_z(tp2, ZC["window"], ZC["min_periods"], ZC["clip"]), tw)
    a, _ = rb.factor_terms_at(z, weights, themes, tp, tw, raw, IDX[-127], ZC)
    b, _ = rb.factor_terms_at(z2, weights, themes, tp2, tw, raw2, IDX[-127], ZC)
    pd.testing.assert_series_equal(a, b)


def test_theme_terms_at_sum_exactly_to_composite():
    z, weights, themes, tw = _setup()
    from ust_rv.features.zscore import rolling_z
    from ust_rv.scoring.composite import theme_pressure

    tz = rolling_z(theme_pressure(z, weights, themes), ZC["window"], ZC["min_periods"], ZC["clip"])
    raw = weighted_average(tz, tw)
    comp = rolling_z(raw, ZC["window"], ZC["min_periods"], ZC["clip"])
    for date in (IDX[-1], IDX[-64]):
        terms, k = rb.theme_terms_at(tz, tw, raw, date, ZC)
        assert terms.sum() + k["composite_mean"] == pytest.approx(float(comp.loc[date]), abs=1e-9)
