import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from ust_rv.models import regime_hmm as rh


def _regime_data(n=3000, seed=0):
    """Two-feature series switching between calm, normal and stressed regimes."""
    rng = np.random.default_rng(seed)
    states = np.repeat(rng.integers(0, 3, n // 100 + 1), 100)[:n]
    means = np.array([[-1.0, 0.5], [0.0, 0.0], [2.0, -1.0]])
    X = means[states] + rng.normal(scale=0.3, size=(n, 2))
    return X, states


def test_fit_labels_sorted_by_vol_feature():
    X, _ = _regime_data()
    f = rh.fit(X, restarts=5)  # spec restarts: seeds 0-1 hit local optima on this data
    assert np.all(np.diff(f.means[:, 0]) > 0)
    assert f.transmat.shape == (3, 3) and np.allclose(f.transmat.sum(axis=1), 1)
    assert np.all(f.expected_duration > 1)


def test_filtered_probs_sum_to_one_and_track_states():
    X, states = _regime_data()
    f = rh.fit(X, restarts=5)  # spec restarts: seeds 0-1 hit local optima on this data
    p = rh.filtered_probs(f, X)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1) == states).mean() > 0.95


def test_filtered_probs_ignore_future_data():
    X, _ = _regime_data()
    f = rh.fit(X, restarts=1)
    t = 2000
    X2 = X.copy()
    X2[t + 1:] = X2[t + 1:][::-1] + 5.0  # rewrite the future
    np.testing.assert_allclose(rh.filtered_probs(f, X)[: t + 1], rh.filtered_probs(f, X2)[: t + 1])


def test_filtered_equals_hmmlearn_smoothed_at_last_observation():
    """Forward-backward and forward-only agree on the final row only: a check that ours is the forward pass."""
    X, _ = _regime_data(n=800)
    m = GaussianHMM(n_components=3, covariance_type="full", n_iter=50, random_state=0).fit(X)
    f = rh.RegimeFit(m.startprob_, m.transmat_, m.means_, m.covars_, 0.0)
    ours = rh.filtered_probs(f, X)
    smoothed = m.predict_proba(X)
    np.testing.assert_allclose(ours[-1], smoothed[-1], atol=1e-8)
    assert not np.allclose(ours[:-1], smoothed[:-1])  # earlier rows differ: smoothing uses the future


def test_walk_forward_refits_each_january_and_is_causal():
    idx = pd.bdate_range("2000-01-03", "2012-12-31")
    X, _ = _regime_data(n=len(idx))
    feat = pd.DataFrame(X, index=idx, columns=["rv21", "slope"])
    wf = rh.walk_forward(feat, restarts=1)
    assert all(d.month == 1 for d in wf.fits)
    first = min(wf.fits)
    assert wf.probs.loc[: first - pd.Timedelta(days=1)].isna().all().all()
    assert np.allclose(wf.probs.dropna().sum(axis=1), 1.0)
    assert min(wf.fits).year == 2008  # 8 years of features before the first fit


def test_cached_fit_reuses_identical_training_data(tmp_path):
    X, _ = _regime_data(n=600)
    f1 = rh.cached_fit(X, 1, tmp_path)
    assert len(list(tmp_path.glob("hmm_*.npz"))) == 1
    f2 = rh.cached_fit(X, 1, tmp_path)
    np.testing.assert_array_equal(f1.transmat, f2.transmat)
    rh.cached_fit(X[:-1], 1, tmp_path)  # different data -> new fit and new file
    assert len(list(tmp_path.glob("hmm_*.npz"))) == 2
