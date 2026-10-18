import numpy as np
import pandas as pd
import pytest

from ust_rv.models.fair_value import rolling_fit

IDX = pd.bdate_range("2000-01-03", periods=600)


def _synthetic(seed=0, beta=(1.0, 0.8, 0.5, -0.2, 0.05), noise=0.01):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(len(IDX), 4)).cumsum(axis=0) * 0.1, index=IDX, columns=["ff", "core_pce_12m", "real_gdp_yoy", "walcl_pct_gdp"])
    y = beta[0] + X.to_numpy() @ np.array(beta[1:]) + rng.normal(scale=noise, size=len(IDX))
    return pd.Series(y, index=IDX), X


def test_recovers_known_coefficients():
    y, X = _synthetic()
    _, _, fits = rolling_fit(y, X, window=250, min_periods=120)
    last = fits.iloc[-1]
    assert last["const"] == pytest.approx(1.0, abs=0.02)
    assert last[["ff", "core_pce_12m", "real_gdp_yoy", "walcl_pct_gdp"]].to_numpy() == pytest.approx([0.8, 0.5, -0.2, 0.05], abs=0.02)
    assert last["r2"] > 0.99


def test_fits_are_monthly_and_respect_min_periods():
    y, X = _synthetic()
    _, _, fits = rolling_fit(y, X, window=250, min_periods=120)
    assert fits.index.to_period("M").is_unique
    assert (fits["nobs"] >= 120).all() and (fits["nobs"] <= 250).all()
    assert fits.index[0] >= IDX[119]


def test_residual_uses_only_past_fits():
    """Shocking y after date t must not change the residual on or before t."""
    y, X = _synthetic()
    resid, _, _ = rolling_fit(y, X, window=250, min_periods=120)
    t = IDX[400]
    y2 = y.copy()
    y2.loc[t + pd.Timedelta(days=1):] += 5.0
    resid2, _, _ = rolling_fit(y2, X, window=250, min_periods=120)
    pd.testing.assert_series_equal(resid.loc[:t], resid2.loc[:t])
    assert not np.allclose(resid.loc[t:].iloc[30:], resid2.loc[t:].iloc[30:])


def test_residual_nan_before_first_fit():
    y, X = _synthetic()
    resid, _, fits = rolling_fit(y, X, window=250, min_periods=120)
    assert resid.loc[: fits.index[0] - pd.Timedelta(days=1)].isna().all()
