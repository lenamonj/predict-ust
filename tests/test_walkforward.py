import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest import walkforward as wf
from ust_rv.models import weights_ridge

IDX = pd.bdate_range("2000-01-03", "2014-12-31")
H, E = 126, 21


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    z = pd.DataFrame(rng.normal(size=(len(IDX), 3)), index=IDX, columns=["a", "b", "c"]).rolling(20).mean().bfill()
    target = 2.0 * z["a"] + 1.0 * z["b"] - 1.0 * z["c"] + rng.normal(scale=0.2, size=len(IDX))
    target.iloc[-H:] = np.nan  # forward targets not yet realized
    return z, target


def test_refit_dates_are_first_january_grid_days():
    d = wf.january_refit_dates(IDX)
    assert all(x.month == 1 for x in d)
    s = IDX.to_series()
    assert all(x == s[s.index.year == x.year].iloc[0] for x in d)


def test_training_ends_147_grid_days_before_refit_and_windows_cover_oos():
    z, target = _synthetic()
    r = wf.run(z, target, H, E, min_train_days=8 * 252)
    for refit, end in r.train_end.items():
        assert IDX.get_loc(refit) - IDX.get_loc(end) >= H + E
    assert r.weights.index[0].year == 2009  # 8y of rows (2016 days) plus the 147-day purge
    oos = r.composite.dropna()
    assert oos.index[0] == r.weights.index[0]
    assert oos.index[-1] == IDX[-1]  # last test window runs to the end


def test_coefficients_non_negative_and_recover_signal():
    z, target = _synthetic()
    r = wf.run(z, target, H, E, min_train_days=8 * 252)
    assert (r.weights >= 0).all().all()
    last = r.weights.iloc[-1]
    assert last["a"] > last["b"] > 0
    assert last["c"] == pytest.approx(0.0, abs=1e-9)  # wrong-signed theme pinned at zero
    assert r.normalized_weights.sum(axis=1).round(9).eq(1.0).all()


def test_no_future_target_leaks_into_weights():
    z, target = _synthetic()
    r1 = wf.run(z, target, H, E, min_train_days=8 * 252)
    t2 = target.copy()
    refit = r1.weights.index[2]
    t2.loc[IDX[IDX.get_loc(refit) - H - E + 1]:] = 99.0  # corrupt everything not allowed in training
    r2 = wf.run(z, t2, H, E, min_train_days=8 * 252)
    pd.testing.assert_series_equal(r1.weights.loc[refit], r2.weights.loc[refit])


def test_standardized_composite_is_trailing_only():
    from ust_rv.features.zscore import rolling_z

    z, target = _synthetic()
    std = lambda c: rolling_z(c, 1260, 252, 3.0)  # noqa: E731
    r1 = wf.run(z, target, H, E, min_train_days=8 * 252, standardize=std)
    t = r1.weights.index[3] + pd.Timedelta(days=100)
    z2 = z.copy()
    z2.loc[t + pd.Timedelta(days=1):] = 50.0  # corrupt the future of the features
    r2 = wf.run(z2, target, H, E, min_train_days=8 * 252, standardize=std)
    pd.testing.assert_series_equal(r1.composite.loc[:t], r2.composite.loc[:t])
    assert 0.6 < r1.composite.dropna().std() < 1.4


def test_apply_weights_all_zero_means_no_view():
    z = pd.DataFrame({"a": [1.0, 2.0]}, index=IDX[:2])
    w = pd.DataFrame({"a": [0.0]}, index=IDX[:1])
    assert wf.apply_weights(z, w).tolist() == [0.0, 0.0]


def test_choose_alpha_returns_candidate():
    z, target = _synthetic()
    mask = target.notna()
    assert weights_ridge.choose_alpha(z[mask], target[mask]) in weights_ridge.ALPHAS


class _Regime:
    """Stand-in for RegimeWalkForward: the same filtered history for every refit."""

    def __init__(self, probs, refits):
        self.history = {r: probs for r in refits}


def _regime_setup():
    from ust_rv.features.zscore import rolling_z

    z, target = _synthetic()
    std = lambda c: rolling_z(c, 1260, 252, 3.0)  # noqa: E731
    g = wf.run(z, target, H, E, min_train_days=8 * 252, standardize=std)
    rng = np.random.default_rng(5)
    raw = pd.DataFrame(rng.uniform(0.1, 1.0, size=(len(IDX), 3)), index=IDX, columns=["low", "mid", "high"])
    probs = raw.div(raw.sum(axis=1), axis=0)
    return z, target, std, g, probs


def test_regime_weights_are_shrunk_halfway_to_global():
    z, target, std, g, probs = _regime_setup()
    rc = wf.run_regime(z, target, g, _Regime(probs, g.weights.index), std, H, E)
    refit = g.weights.index[2]
    cutoff = IDX[IDX.get_loc(refit) - H - E]
    mask = target.notna() & (IDX <= cutoff)
    f = weights_ridge.fit_ridge(z.fillna(0)[mask], target[mask], g.alpha[refit], sample_weight=probs.loc[mask, "high"])
    expected = 0.5 * f.coef + 0.5 * g.weights.loc[refit]
    pd.testing.assert_series_equal(rc.weights[refit].loc["high"], expected, check_names=False)
    assert (rc.weights[refit] >= 0).all().all()


def test_regime_composite_blends_by_probability_and_is_causal():
    z, target, std, g, probs = _regime_setup()
    rc = wf.run_regime(z, target, g, _Regime(probs, g.weights.index), std, H, E)
    assert rc.composite.loc[g.weights.index[0]:].notna().all()
    t = g.weights.index[4] + pd.Timedelta(days=60)
    z2, t2 = z.copy(), target.copy()
    z2.loc[t + pd.Timedelta(days=1):] = 9.0
    t2.loc[IDX[IDX.get_loc(g.weights.index[4]) - H - E + 1]:] = 99.0
    g2 = wf.run(z2, t2, H, E, min_train_days=8 * 252, standardize=std)
    rc2 = wf.run_regime(z2, t2, g2, _Regime(probs, g2.weights.index), std, H, E)
    pd.testing.assert_series_equal(rc.composite.loc[:t], rc2.composite.loc[:t])
