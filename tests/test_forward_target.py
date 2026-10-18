import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest import forward_target as ft

IDX = pd.bdate_range("2020-01-01", periods=300)


def _curve(y6m, y2, y3, y10, y20):
    return pd.DataFrame({"y6m": y6m, "y2": y2, "y3": y3, "y10": y10, "y20": y20}, index=IDX)


def test_flat_curve_forward_equals_spot():
    c = _curve(4.0, 4.0, 4.0, 4.0, 4.0)
    assert np.allclose(ft.forward_yield(c, 2.0), 4.0) and np.allclose(ft.forward_yield(c, 10.0), 4.0)


def test_upward_curve_forward_above_spot():
    c = _curve(3.0, 3.5, 3.7, 4.0, 4.5)
    f2 = ft.forward_yield(c, 2.0).iloc[0]
    assert f2 == pytest.approx((2.5 * 3.6 - 0.5 * 3.0) / 2.0)
    assert f2 > 3.5
    assert ft.forward_yield(c, 10.0).iloc[0] > 4.0


def test_forward_target_is_realized_minus_forward_and_nan_until_realized():
    y2 = np.linspace(4.0, 5.0, len(IDX))
    c = _curve(4.0, y2, y2, 4.0, 4.0)
    t = ft.forward_target(c, "2y", 126)
    expected0 = (y2[126] - ft.forward_yield(c, 2.0).iloc[0]) * 100.0
    assert t.iloc[0] == pytest.approx(expected0)
    assert t.iloc[-126:].isna().all() and t.iloc[:-126].notna().all()
