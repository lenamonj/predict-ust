import numpy as np
import pandas as pd
import pytest

from ust_rv.backtest.walkforward import january_refit_dates
from ust_rv.models import challenger_gbm as cg

IDX = pd.bdate_range("2000-01-03", "2014-12-31")
H, E = 126, 21


class _Regime:
    def __init__(self, probs, refits):
        self.history = {r: probs for r in refits}


def _setup():
    rng = np.random.default_rng(0)
    tz = pd.DataFrame(rng.normal(size=(len(IDX), 2)), index=IDX, columns=["growth", "fed"]).rolling(20).mean().bfill()
    pressure = pd.DataFrame(rng.normal(size=(len(IDX), 3)), index=IDX, columns=["A1", "B1", "C1"])
    probs = pd.DataFrame(1 / 3, index=IDX, columns=["low", "mid", "high"])
    target = 2 * tz["growth"] - tz["fed"] + rng.normal(scale=0.3, size=len(IDX))
    target.iloc[-H:] = np.nan
    refits = january_refit_dates(IDX)[-4:]
    return tz, pressure, probs, target, refits


def test_top_factors_by_effective_weight_with_id_ties():
    lib = [
        {"id": "B2", "theme": "labor", "weight_2y": 1.0}, {"id": "A1", "theme": "growth", "weight_2y": 1.0},
        {"id": "C1", "theme": "inflation", "weight_2y": 1.0}, {"id": "E1", "theme": "supply", "weight_2y": 0.0},
        {"id": "B1", "theme": "labor", "weight_2y": 1.0},
    ]
    tw = pd.Series({"labor": 1.25, "growth": 1.0, "inflation": 1.5, "supply": 1.0})
    assert cg.top_factors(lib, tw, "2y", {"A1", "B1", "B2", "C1", "E1"}, n=3) == ["C1", "B1", "B2"]
    assert "B1" not in cg.top_factors(lib, tw, "2y", {"A1", "B2", "C1"}, n=3)  # unavailable factors skipped


def test_run_predicts_out_of_sample_only_and_standardizes():
    tz, pressure, probs, target, refits = _setup()
    r = cg.run(tz, pressure, ["A1", "B1"], _Regime(probs, refits), target, refits, H, E, 0.5, 0.2)
    assert r.pred_z.loc[: refits[0] - pd.Timedelta(days=1)].isna().all()
    assert r.pred_z.loc[refits[0]:].notna().all()
    assert (r.n_trees >= 1).all()
    assert 0.5 < r.pred_z.dropna().std() < 2.0
    assert np.corrcoef(r.pred_z.dropna(), (2 * tz["growth"] - tz["fed"]).loc[r.pred_z.dropna().index])[0, 1] > 0.8


def test_training_ignores_future_targets():
    tz, pressure, probs, target, refits = _setup()
    r1 = cg.run(tz, pressure, ["A1"], _Regime(probs, refits), target, refits[:2], H, E, 0.5, 0.2)
    t2 = target.copy()
    t2.loc[IDX[IDX.get_loc(refits[1]) - H - E + 1]:] = 50.0  # rows the refit at refits[1] must not see
    r2 = cg.run(tz, pressure, ["A1"], _Regime(probs, refits), t2, refits[:2], H, E, 0.5, 0.2)
    pd.testing.assert_series_equal(r1.pred_z, r2.pred_z)


def test_shap_sums_to_raw_prediction():
    tz, pressure, probs, target, refits = _setup()
    r = cg.run(tz, pressure, ["A1"], _Regime(probs, refits), target, refits[-1:], H, E, 0.5, 0.2)
    shap = r.shap_latest()
    raw = r.latest_model.predict(r.latest_features)[0]
    assert shap.sum() == pytest.approx(raw, rel=1e-6)
    assert "bias" in shap.index and "theme_growth" in shap.index


def test_agreement():
    a = pd.Series([1.0, 0.0, -1.0, 1.0])
    b = pd.Series([1.0, 1.0, -1.0, np.nan])
    assert cg.agreement(a, b) == pytest.approx(2 / 3)
