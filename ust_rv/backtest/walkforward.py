"""Purged, embargoed annual walk-forward for the ridge theme weights (Build_Plan 11.1)."""
from dataclasses import dataclass

import pandas as pd

from ust_rv.models import weights_ridge


@dataclass
class WalkForward:
    weights: pd.DataFrame  # refit date x theme (raw ridge coefficients)
    alpha: pd.Series  # refit date -> chosen alpha
    train_end: pd.Series  # refit date -> last training row date
    composite: pd.Series  # out-of-sample composite on the grid (NaN before the first refit)

    @property
    def normalized_weights(self) -> pd.DataFrame:
        total = self.weights.sum(axis=1)
        return self.weights.div(total.where(total > 0), axis=0)


def january_refit_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(s.index.year).first().loc[lambda d: d.dt.month == 1].to_numpy())


def apply_weights(theme_z: pd.DataFrame, weights_by_date: pd.DataFrame) -> pd.Series:
    """sum(w * z) / sum(w) over available themes, weights held from each refit to the next."""
    W = weights_by_date.reindex(theme_z.index, method="ffill")[theme_z.columns]
    num = (theme_z * W).sum(axis=1, min_count=1)
    den = (theme_z.notna() * W).sum(axis=1)
    comp = num / den.where(den > 0)
    # all-zero learned weights: no view (composite 0) rather than undefined
    no_view = W.notna().all(axis=1) & (W.sum(axis=1) == 0)
    return comp.mask(no_view, 0.0).where(W.notna().all(axis=1))


def run(theme_z: pd.DataFrame, target: pd.Series, horizon: int, embargo: int, min_train_days: int,
        alphas=weights_ridge.ALPHAS, standardize=None) -> WalkForward:
    """theme_z: grid x theme (pressure space). target: vol-scaled forward change aligned to t.
    `standardize`: optional trailing transform (e.g. rolling z) applied to the composite."""
    X_all = theme_z.fillna(0.0)
    has_data = theme_z.notna().any(axis=1)
    gap = horizon + embargo
    rows, alphas_used, ends = {}, {}, {}
    for refit in january_refit_dates(theme_z.index):
        pos = theme_z.index.get_loc(refit) - gap
        if pos < 0:
            continue
        cutoff = theme_z.index[pos]
        mask = has_data & target.notna() & (theme_z.index <= cutoff)
        if mask.sum() < min_train_days:
            continue
        f = weights_ridge.fit(X_all[mask], target[mask], alphas, gap=gap)
        rows[refit], alphas_used[refit], ends[refit] = f.coef, f.alpha, theme_z.index[mask][-1]
    weights = pd.DataFrame(rows).T.reindex(columns=theme_z.columns)
    weights.index.name = "refit"
    if weights.empty:
        comp = pd.Series(float("nan"), index=theme_z.index)
    elif standardize is None:
        comp = apply_weights(theme_z, weights)
    else:
        comp = standardized_composite(theme_z, weights, standardize)
    return WalkForward(weights, pd.Series(alphas_used, dtype=float), pd.Series(ends), comp)


def standardized_composite(theme_z: pd.DataFrame, weights: pd.DataFrame, standardize) -> pd.Series:
    """For each test period, apply that refit's weights to the full theme history, standardize with
    trailing windows, and keep the test-period slice. Weights and data are both known at each date."""
    comp = pd.Series(float("nan"), index=theme_z.index)
    bounds = list(weights.index) + [theme_z.index[-1] + pd.Timedelta(days=1)]
    for k, refit in enumerate(weights.index):
        period = (theme_z.index >= refit) & (theme_z.index < bounds[k + 1])
        w = weights.iloc[[k]]
        if w.sum(axis=1).iloc[0] == 0:
            comp[period] = 0.0  # all-zero learned weights: no view
            continue
        raw = apply_weights(theme_z, w.set_axis([theme_z.index[0]]))
        comp[period] = standardize(raw)[period]
    return comp


SHRINK_TO_GLOBAL = 0.5


@dataclass
class RegimeComposite:
    weights: dict  # refit date -> DataFrame(state x theme), after shrinkage
    composite: pd.Series


def run_regime(theme_z: pd.DataFrame, target: pd.Series, global_wf: WalkForward, regime, standardize, horizon: int, embargo: int) -> RegimeComposite:
    """Regime-conditional weights (Build_Plan 8): at each ridge refit, fit one ridge per regime with
    sample weights = filtered state probability (from the HMM fitted at that date), shrink 50% to the
    global weights, and blend the three standardized regime composites by today's filtered probabilities."""
    X_all = theme_z.fillna(0.0)
    has_data = theme_z.notna().any(axis=1)
    gap = horizon + embargo
    comp = pd.Series(float("nan"), index=theme_z.index)
    bounds = list(global_wf.weights.index) + [theme_z.index[-1] + pd.Timedelta(days=1)]
    weights = {}
    for k, refit in enumerate(global_wf.weights.index):
        if refit not in regime.history:
            continue
        probs = regime.history[refit].reindex(theme_z.index)
        cutoff = theme_z.index[theme_z.index.get_loc(refit) - gap]
        mask = has_data & target.notna() & (theme_z.index <= cutoff) & probs.notna().all(axis=1)
        g = global_wf.weights.loc[refit]
        per_state = {}
        for state in probs.columns:
            f = weights_ridge.fit_ridge(X_all[mask], target[mask], global_wf.alpha[refit], sample_weight=probs.loc[mask, state])
            per_state[state] = SHRINK_TO_GLOBAL * f.coef + (1 - SHRINK_TO_GLOBAL) * g
        W = pd.DataFrame(per_state).T
        weights[refit] = W
        period = (theme_z.index >= refit) & (theme_z.index < bounds[k + 1])
        blend = pd.Series(0.0, index=theme_z.index[period])
        for state in W.index:
            w = W.loc[[state]]
            if w.sum(axis=1).iloc[0] == 0:
                c = pd.Series(0.0, index=theme_z.index)
            else:
                c = standardize(apply_weights(theme_z, w.set_axis([theme_z.index[0]])))
            blend += probs.loc[period, state] * c[period]
        comp[period] = blend
    return RegimeComposite(weights, comp)
