"""Regime model (Build_Plan Section 8): 3-state Gaussian HMM on market-stress features.

Only FILTERED probabilities are used: p(state_t | x_1..x_t), from a normalized forward
recursion written here. hmmlearn's predict_proba runs forward-backward (smoothed, uses
future data) and is never called by this module."""
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

from ust_rv.backtest.walkforward import january_refit_dates
from ust_rv.features.zscore import rolling_z

N_STATES = 3
LABELS = ["low", "mid", "high"]
FEATURES = ["rv21", "slope", "eq_bond_corr60", "credit_chg21"]
MIN_FIT_DAYS = 8 * 252

logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # per-restart convergence chatter


@dataclass
class RegimeFit:
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    loglik: float

    @property
    def expected_duration(self) -> np.ndarray:
        return 1.0 / (1.0 - np.diag(self.transmat))


def features(market: pd.DataFrame, window: int, min_periods: int, clip: float) -> pd.DataFrame:
    """market: grid frame with y2, y10 (%), equity (index level), baa10y (%)."""
    dy10 = market["y10"].diff() * 100.0
    raw = pd.DataFrame(
        {
            "rv21": dy10.rolling(21).std(),
            "slope": (market["y10"] - market["y2"]) * 100.0,
            "eq_bond_corr60": np.log(market["equity"]).diff().rolling(60).corr(-dy10),
            "credit_chg21": market["baa10y"].diff(21) * 100.0,
        }
    )
    return rolling_z(raw, window, min_periods, clip)


def fit(X: np.ndarray, restarts: int = 5, n_iter: int = 200) -> RegimeFit:
    """Best-likelihood fit over seeds 0..restarts-1, states relabeled low/mid/high by mean of feature 0 (vol)."""
    best = None
    for seed in range(restarts):
        m = GaussianHMM(n_components=N_STATES, covariance_type="full", n_iter=n_iter, random_state=seed)
        m.fit(X)
        ll = m.score(X)
        if best is None or ll > best[1]:
            best = (m, ll)
    m, ll = best
    order = np.argsort(m.means_[:, 0])
    return RegimeFit(m.startprob_[order], m.transmat_[np.ix_(order, order)], m.means_[order], m.covars_[order], float(ll))


def filtered_probs(f: RegimeFit, X: np.ndarray) -> np.ndarray:
    """Normalized forward recursion: row t is p(state_t | x_1..x_t)."""
    log_e = np.column_stack([multivariate_normal.logpdf(X, f.means[k], f.covars[k], allow_singular=True) for k in range(N_STATES)])
    e = np.exp(log_e - log_e.max(axis=1, keepdims=True))  # per-row scaling cancels in the normalization
    out = np.empty_like(e)
    alpha = f.startprob * e[0]
    out[0] = alpha / alpha.sum()
    for t in range(1, len(X)):
        alpha = (out[t - 1] @ f.transmat) * e[t]
        out[t] = alpha / alpha.sum()
    return out


@dataclass
class RegimeWalkForward:
    probs: pd.DataFrame  # grid x LABELS, filtered, NaN before the first fit
    fits: dict  # refit date -> RegimeFit
    history: dict  # refit date -> filtered probs over all feature history under that fit (training weights)

    @property
    def latest(self) -> RegimeFit:
        return self.fits[max(self.fits)]


def cached_fit(X: np.ndarray, restarts: int, cache_dir: Path | None) -> RegimeFit:
    """Fits for past refit dates never change (market features are not revised), so they are cached
    by a hash of the exact training data; any data change produces a new key and a fresh fit."""
    if cache_dir is None:
        return fit(X, restarts)
    key = hashlib.sha1(np.ascontiguousarray(X).tobytes() + str(restarts).encode()).hexdigest()[:16]
    path = cache_dir / f"hmm_{key}.npz"
    if path.exists():
        z = np.load(path)
        return RegimeFit(z["startprob"], z["transmat"], z["means"], z["covars"], float(z["loglik"]))
    f = fit(X, restarts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, startprob=f.startprob, transmat=f.transmat, means=f.means, covars=f.covars, loglik=f.loglik)
    return f


def walk_forward(feat: pd.DataFrame, restarts: int = 5, cache_dir: Path | None = None) -> RegimeWalkForward:
    """Refit each January on feature history up to the refit date; filter with that fit until the next."""
    valid = feat.dropna()
    probs = pd.DataFrame(np.nan, index=feat.index, columns=LABELS)
    fits, history = {}, {}
    refits = [d for d in january_refit_dates(feat.index) if (valid.index < d).sum() >= MIN_FIT_DAYS]
    bounds = refits + [feat.index[-1] + pd.Timedelta(days=1)]
    for k, refit in enumerate(refits):
        f = cached_fit(valid.loc[valid.index < refit].to_numpy(), restarts, cache_dir)
        fits[refit] = f
        p = pd.DataFrame(filtered_probs(f, valid.to_numpy()), index=valid.index, columns=LABELS)
        history[refit] = p
        period = (p.index >= refit) & (p.index < bounds[k + 1])
        probs.loc[p.index[period]] = p.loc[period].to_numpy()
    return RegimeWalkForward(probs, fits, history)
