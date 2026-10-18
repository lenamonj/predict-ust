"""Gradient boosting challenger (Build_Plan 9.2). Advisory only: the RV score comes from the composite.

Annual refit on the same dates as the ridge walk-forward. n_estimators by early stopping on the
last 20% of the (purged) training window; the model is then refit on the full training window with
that tree count. Training rows are every 5th grid day: overlapping 6m targets make adjacent days
near-duplicates. Predictions are standardized by the training-fit prediction mean and sd so the
composite's hysteresis thresholds apply."""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from ust_rv.scoring.decision import hysteresis

PARAMS = dict(max_depth=3, learning_rate=0.03, min_child_samples=50, n_estimators=2000, verbose=-1, random_state=0)
PATIENCE = 100
VALID_SHARE = 0.20
ROW_STEP = 5
N_TOP_FACTORS = 10


@dataclass
class ChallengerResult:
    pred_z: pd.Series  # standardized out-of-sample prediction (pressure space: + = yields higher)
    rv: pd.Series
    n_trees: pd.Series  # refit date -> trees chosen by early stopping
    latest_model: lgb.LGBMRegressor
    latest_features: pd.DataFrame  # feature row for the last grid date
    pred_scale: tuple[float, float]  # (mean, sd) of training-fit predictions, latest refit

    def shap_latest(self) -> pd.Series:
        """TreeSHAP contributions for the last date (LightGBM built-in), plus the bias term; sums to the raw prediction."""
        contrib = self.latest_model.predict(self.latest_features, pred_contrib=True)[0]
        return pd.Series(contrib, index=[*self.latest_features.columns, "bias"])


def top_factors(factor_list: list[dict], theme_weights: pd.Series, tenor: str, available: set[str], n: int = N_TOP_FACTORS) -> list[str]:
    """Highest effective weight in the composite (factor weight x theme weight), ties broken by id."""
    rows = [(f[f"weight_{tenor}"] * theme_weights[f["theme"]], f["id"]) for f in factor_list
            if f[f"weight_{tenor}"] > 0 and f["id"] in available]
    return [fid for _, fid in sorted(rows, key=lambda r: (-r[0], r[1]))[:n]]


def feature_frame(theme_z: pd.DataFrame, pressure: pd.DataFrame, factor_ids: list[str], regime_probs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([theme_z.add_prefix("theme_"), regime_probs.add_prefix("regime_"), pressure[factor_ids].add_prefix("f_")], axis=1)


def _fit(X: pd.DataFrame, y: pd.Series, gap: int) -> tuple[lgb.LGBMRegressor, int]:
    n_val = max(int(len(X) * VALID_SHARE), 1)
    fit_end = len(X) - n_val - max(gap // ROW_STEP, 1)  # purge in sampled rows
    es = lgb.LGBMRegressor(**PARAMS).fit(
        X.iloc[:fit_end], y.iloc[:fit_end], eval_X=X.iloc[-n_val:], eval_y=y.iloc[-n_val:],
        callbacks=[lgb.early_stopping(PATIENCE, verbose=False)],
    )
    n_trees = max(int(es.best_iteration_ or PATIENCE), 1)
    final = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": n_trees}).fit(X, y)
    return final, n_trees


def run(theme_z, pressure, factor_ids, regime, target: pd.Series, refits: pd.DatetimeIndex, horizon: int, embargo: int,
        enter: float, exit_: float) -> ChallengerResult:
    """regime: RegimeWalkForward (training rows use the regime fit made at each refit date)."""
    gap = horizon + embargo
    index = theme_z.index
    pred_z = pd.Series(float("nan"), index=index)
    n_trees = {}
    bounds = list(refits) + [index[-1] + pd.Timedelta(days=1)]
    model = X = scale = None
    for k, refit in enumerate(refits):
        probs = regime.history[refit].reindex(index)
        X = feature_frame(theme_z, pressure, factor_ids, probs)
        cutoff = index[index.get_loc(refit) - gap]
        rows = X.index[(target.notna() & (index <= cutoff) & theme_z.notna().any(axis=1)).to_numpy()][::ROW_STEP]
        model, n_trees[refit] = _fit(X.loc[rows], target.loc[rows], gap)
        fitted = model.predict(X.loc[rows])
        scale = (float(fitted.mean()), float(fitted.std()))
        period = (index >= refit) & (index < bounds[k + 1])
        pred_z[period] = (model.predict(X.loc[period]) - scale[0]) / scale[1]
    rv = hysteresis(pred_z, enter, exit_)
    return ChallengerResult(pred_z, rv, pd.Series(n_trees), model, X.iloc[[-1]], scale)


def agreement(challenger_rv: pd.Series, composite_rv: pd.Series) -> float:
    pairs = pd.concat([challenger_rv, composite_rv], axis=1, sort=True).dropna()
    return float((pairs.iloc[:, 0] == pairs.iloc[:, 1]).mean()) if len(pairs) else float("nan")
