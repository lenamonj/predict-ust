import numpy as np
import pandas as pd
import pytest

from ust_rv.features.zscore import rolling_z
from ust_rv.scoring.composite import theme_pressure, weighted_average
from ust_rv.scoring.decision import curve_read, hysteresis, next_threshold
from ust_rv.scoring.scorecard import forward_change_bp, sign_check

IDX = pd.bdate_range("2020-01-01", periods=40)


# ---------- z-score ----------

def test_rolling_z_trailing_min_history_and_clip():
    x = pd.Series(np.arange(40, dtype=float), index=IDX)
    x.iloc[-1] = 1e6
    # an outlier inside its own 10-obs window has z = 9/sqrt(10) = 2.85, so test the clip at 2
    z = rolling_z(x, window=10, min_periods=5, clip=2.0)
    assert z.iloc[:4].isna().all() and z.iloc[4:].notna().all()
    assert z.iloc[-1] == 2.0
    # trailing only: changing the future does not change the past
    x2 = x.copy()
    x2.iloc[30:] = -50
    pd.testing.assert_series_equal(rolling_z(x2, 10, 5, 2.0).iloc[:30], z.iloc[:30])
    window = x.iloc[5:15]
    assert z.iloc[14] == pytest.approx((x.iloc[14] - window.mean()) / window.std())


def test_rolling_z_constant_series_is_nan():
    z = rolling_z(pd.Series(1.0, index=IDX), 10, 5, 3.0)
    assert z.isna().all()


# ---------- theme aggregation and composite ----------

def test_theme_pressure_renormalizes_over_available_factors():
    p = pd.DataFrame({"a": [1.0, 1.0], "b": [-1.0, np.nan], "c": [2.0, 2.0]}, index=IDX[:2])
    w = pd.Series({"a": 1.0, "b": 3.0, "c": 0.0})
    themes = pd.Series({"a": "t1", "b": "t1", "c": "t2"})
    tp = theme_pressure(p, w, themes)
    assert list(tp.columns) == ["t1"]  # t2 has zero weight for this tenor
    assert tp.loc[IDX[0], "t1"] == pytest.approx((1 - 3) / 4)
    assert tp.loc[IDX[1], "t1"] == pytest.approx(1.0)  # b missing -> renormalized over a


def test_weighted_average_all_missing_is_nan():
    v = pd.DataFrame({"a": [np.nan]}, index=IDX[:1])
    assert np.isnan(weighted_average(v, pd.Series({"a": 1.0})).iloc[0])


def test_composite_weights():
    tz = pd.DataFrame({"growth": [1.0], "inflation": [-1.0]}, index=IDX[:1])
    comp = weighted_average(tz, pd.Series({"growth": 1.0, "inflation": 1.5}))
    assert comp.iloc[0] == pytest.approx((1.0 - 1.5) / 2.5)


# ---------- decision ----------

def test_hysteresis_scripted_path_and_sign_convention():
    p = pd.Series([np.nan, 0.1, 0.4, 0.55, 0.3, 0.2, -0.3, -0.6, -0.4, np.nan, 0.6], index=IDX[:11])
    rv = hysteresis(p, enter=0.5, exit=0.25)
    expected = [np.nan, 0, 0, -1, -1, 0, 0, 1, 1, 1, -1]
    np.testing.assert_array_equal(rv.to_numpy(), np.array(expected, dtype=float))


def test_strong_negative_pressure_maps_to_buy():
    assert hysteresis(pd.Series([-2.0]), 0.5, 0.25).iloc[0] == 1.0


def test_curve_read():
    assert curve_read(1, 0) == "steepener"
    assert curve_read(0, 1) == "flattener"
    assert curve_read(-1, -1) == "neutral"


def test_next_threshold():
    assert next_threshold(0.3, 0, 0.5, 0.25)[1] == pytest.approx(0.2)
    assert next_threshold(-0.1, 0, 0.5, 0.25)[1] == pytest.approx(-0.4)
    assert next_threshold(0.6, -1, 0.5, 0.25)[1] == pytest.approx(-0.35)


# ---------- target and sign check ----------

def test_forward_change_bp():
    y = pd.DataFrame({"y10": [4.0, 4.1, 4.3]}, index=IDX[:3])
    dy = forward_change_bp(y, 2)
    assert dy["y10"].iloc[0] == pytest.approx(30.0)
    assert dy["y10"].iloc[1:].isna().all()


def test_sign_check_uses_only_realized_targets():
    n, h = 300, 20
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(1)
    pressure = pd.Series(rng.normal(size=n), index=idx)
    dy = pressure * 10 + rng.normal(size=n)
    assert sign_check(pressure, dy, h, 200) > 0.9
    assert sign_check(-pressure, dy, h, 200) < -0.9
    # garbage in the unrealized tail must not matter
    dy_tail = dy.copy()
    dy_tail.iloc[-h:] = -1e9
    assert sign_check(pressure, dy_tail, h, 200) == pytest.approx(sign_check(pressure, dy, h, 200))
