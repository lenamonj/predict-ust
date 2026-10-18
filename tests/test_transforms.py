import numpy as np
import pandas as pd
import pytest

from ust_rv.features.transforms import TRANSFORMS, apply_transform

IDX = pd.date_range("2020-01-01", periods=8, freq="MS")
S = pd.Series([100.0, 101.0, 103.0, 102.0, 104.0, 108.0, 110.0, 111.0], index=IDX)


def test_level():
    pd.testing.assert_series_equal(apply_transform(S, {"type": "level"}), S)


def test_diff():
    out = apply_transform(S, {"type": "diff", "periods": 3})
    assert out.iloc[3] == pytest.approx(2.0)
    assert out.iloc[7] == pytest.approx(111 - 104)
    assert out.iloc[:3].isna().all()


def test_pct_change():
    out = apply_transform(S, {"type": "pct_change", "periods": 2})
    assert out.iloc[2] == pytest.approx(3.0)


def test_pct_change_annualized():
    out = apply_transform(S, {"type": "pct_change_annualized", "periods": 3, "ppy": 12})
    assert out.iloc[3] == pytest.approx(((102 / 100) ** 4 - 1) * 100)


def test_mean_change():
    out = apply_transform(S, {"type": "mean_change", "periods": 3})
    assert out.iloc[5] == pytest.approx((108 - 103) / 3)


def test_accel():
    s = pd.Series(np.linspace(100, 130, 30), index=pd.date_range("2020-01-01", periods=30, freq="MS"))
    out = apply_transform(s, {"type": "accel", "periods": 3, "ppy": 12})
    a3 = ((s.iloc[29] / s.iloc[26]) ** 4 - 1) * 100
    yoy = (s.iloc[29] / s.iloc[17] - 1) * 100
    assert out.iloc[29] == pytest.approx(a3 - yoy)


def test_diff_vs_ma():
    out = apply_transform(S, {"type": "diff_vs_ma", "window": 4})
    assert out.iloc[3] == pytest.approx(102 - (100 + 101 + 103 + 102) / 4)


def test_momentum_12m_1m():
    out = apply_transform(S, {"type": "momentum_12m_1m", "long": 4, "short": 1})
    # change from t-4 to t-1
    assert out.iloc[7] == pytest.approx(110 - 102)


def test_roll_and_scale():
    out = apply_transform(S, {"type": "diff", "periods": 1, "roll": 2, "roll_how": "sum", "scale": 100})
    assert out.iloc[2] == pytest.approx(((101 + 103) - (100 + 101)) * 100)


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        apply_transform(S, {"type": "nope"})


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "level"},
        {"type": "diff", "periods": 3},
        {"type": "pct_change", "periods": 2},
        {"type": "pct_change_annualized", "periods": 3, "ppy": 12},
        {"type": "mean_change", "periods": 3},
        {"type": "accel", "periods": 3, "ppy": 4},
        {"type": "diff_vs_ma", "window": 3},
        {"type": "momentum_12m_1m", "long": 4, "short": 1},
    ],
)
def test_trailing_only(spec):
    """Appending future observations must not change past outputs."""
    full = apply_transform(S, spec)
    trunc = apply_transform(S.iloc[:5], spec)
    pd.testing.assert_series_equal(full.iloc[:5], trunc)


def test_every_transform_has_trailing_test():
    tested = {"level", "diff", "pct_change", "pct_change_annualized", "mean_change", "accel", "diff_vs_ma", "momentum_12m_1m"}
    assert tested == set(TRANSFORMS)
