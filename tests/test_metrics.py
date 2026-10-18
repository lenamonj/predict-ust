import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest import metrics as m

IDX = pd.bdate_range("2010-01-01", periods=2000)


def test_ic_and_hit_rate_signs():
    rng = np.random.default_rng(0)
    comp = pd.Series(rng.normal(size=len(IDX)), index=IDX)
    dy = comp * 20 + rng.normal(size=len(IDX))
    assert m.ic(comp, dy) > 0.9
    rv = -np.sign(comp)  # positive pressure -> sell
    assert m.hit_rate(rv, dy) > 0.95
    assert m.hit_rate(-rv, dy) < 0.05
    assert np.isnan(m.hit_rate(pd.Series(0.0, index=IDX), dy))


def test_neutral_share_hold_turnover():
    rv = pd.Series([0, 1, 1, 1, 0, 0, -1, -1, 0, 0], index=pd.bdate_range("2020-01-01", periods=10), dtype=float)
    assert m.neutral_share(rv) == pytest.approx(0.5)
    assert m.avg_hold(rv) == pytest.approx((3 + 2) / 2)
    assert m.turnover(rv) == pytest.approx(4 / ((rv.index[-1] - rv.index[0]).days / 365.25))


def test_max_adverse():
    rv = pd.Series([1.0, -1.0, 0.0], index=IDX[:3])
    dy = pd.Series([30.0, 10.0, 500.0], index=IDX[:3])  # long hurt by +30, short helped by +10, flat ignored
    assert m.max_adverse(rv, dy) == pytest.approx(30.0)


def test_overlay_pnl_direction_carry_and_lag():
    idx = pd.bdate_range("2020-01-06", periods=3)  # Mon, Tue, Wed
    y = pd.Series([4.00, 3.90, 3.90], index=idx)
    flat = pd.Series([4.00, 3.90, 3.90], index=idx)
    fin = pd.Series([4.00, 3.90, 3.90], index=idx)  # zero carry
    rv = pd.Series([1.0, 1.0, 1.0], index=idx)
    pnl = m.overlay_pnl(rv, y, flat, 1.0, 2.0, fin)
    assert pnl.iloc[0] == 0.0  # no position before the first signal
    assert pnl.iloc[1] == pytest.approx(10.0)  # long gains 10bp when yield falls 10bp
    # positive carry for a long when yield > financing
    fin2 = pd.Series(2.0, index=idx)
    pnl2 = m.overlay_pnl(rv, pd.Series(4.0, index=idx), pd.Series(4.0, index=idx), 1.0, 2.0, fin2)
    d = m.par_duration(pd.Series([4.0]), 2.0).iloc[0]
    assert pnl2.iloc[1] == pytest.approx(200.0 / d / 360.0)
    # roll: upward slope pays a long
    pnl3 = m.overlay_pnl(rv, pd.Series(4.0, index=idx), pd.Series(3.0, index=idx), 1.0, 2.0, pd.Series(4.0, index=idx))
    assert pnl3.iloc[1] == pytest.approx(100.0 / 365.0)


def test_par_duration():
    d = m.par_duration(pd.Series([5.0, 0.0]), 10.0)
    assert 7.7 < d.iloc[0] < 7.9
    assert d.iloc[1] == 10.0


def test_sharpe_monthly():
    pnl = pd.Series(np.tile([1.0, 2.0], len(IDX) // 2), index=IDX)
    assert m.sharpe(pnl) > 10


def test_block_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(1)
    f = pd.DataFrame({"x": rng.normal(size=1000)})
    lo, hi = m.block_bootstrap(f, lambda d: d["x"].mean(), block=50, n_draws=300)
    assert lo < f["x"].mean() < hi


def test_hac_tstat_smaller_than_ols_on_overlapping_targets():
    rng = np.random.default_rng(2)
    shocks = pd.Series(rng.normal(size=len(IDX)), index=IDX)
    y = shocks.rolling(126).sum()  # overlapping 6m target
    x = shocks.rolling(126).sum().shift(126) + rng.normal(size=len(IDX))
    r = m.hac_regression(x, y, lags=126)
    assert abs(r["t_hac"]) < abs(r["t_ols"])


def test_calibration_monotonic_on_clean_signal():
    rng = np.random.default_rng(3)
    comp = pd.Series(rng.normal(size=len(IDX)), index=IDX)
    cal = m.calibration(comp, comp * 10)
    assert len(cal) == 10 and cal["mean_dy_bp"].is_monotonic_increasing


def test_realized_vol_6m():
    y = pd.Series(np.r_[0, np.cumsum(np.tile([0.01, -0.01], 100))], index=pd.bdate_range("2020-01-01", periods=201))
    rv = m.realized_vol_6m(y)
    assert rv.dropna().iloc[-1] == pytest.approx(pd.Series(np.tile([1.0, -1.0], 63)).std() * np.sqrt(126), rel=1e-6)
