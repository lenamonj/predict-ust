"""Learned theme weights (Build_Plan 9.1): non-negative ridge of the vol-scaled forward yield
change on the theme z's. Positivity keeps every theme's sign as constructed."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Build_Plan grid {0.1, 1, 10, 100} extended: CV chose the boundary (100) in most refits.
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)


@dataclass
class RidgeFit:
    alpha: float
    coef: pd.Series
    intercept: float


def fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float, sample_weight: pd.Series | None = None) -> RidgeFit:
    sw = None if sample_weight is None else sample_weight.to_numpy()
    model = Ridge(alpha=alpha, positive=True).fit(X.to_numpy(), y.to_numpy(), sample_weight=sw)
    return RidgeFit(alpha, pd.Series(model.coef_, index=X.columns), float(model.intercept_))


def choose_alpha(X: pd.DataFrame, y: pd.Series, alphas=ALPHAS, n_folds: int = 3, gap: int = 147) -> float:
    """Expanding time-series CV: fold k trains on chunks < k (less a `gap` purge) and validates on chunk k."""
    edges = np.linspace(0, len(X), n_folds + 2).astype(int)
    scores = {a: [] for a in alphas}
    for k in range(1, n_folds + 1):
        train_end, val_end = edges[k] - gap, edges[k + 1]
        if train_end <= 0:
            continue
        Xt, yt, Xv, yv = X.iloc[:train_end], y.iloc[:train_end], X.iloc[edges[k]:val_end], y.iloc[edges[k]:val_end]
        for a in alphas:
            fit = fit_ridge(Xt, yt, a)
            pred = Xv.to_numpy() @ fit.coef.to_numpy() + fit.intercept
            scores[a].append(float(np.mean((yv.to_numpy() - pred) ** 2)))
    if not any(scores.values()):
        raise ValueError("not enough rows for alpha cross-validation")
    return min(alphas, key=lambda a: np.mean(scores[a]))


def fit(X: pd.DataFrame, y: pd.Series, alphas=ALPHAS, gap: int = 147) -> RidgeFit:
    return fit_ridge(X, y, choose_alpha(X, y, alphas, gap=gap))
