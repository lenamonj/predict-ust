"""10y fair value model (factor F3).

DGS10 = a + b1*DFF + b2*core_PCE_12m + b3*real_GDP_yoy + b4*WALCL/GDP + e

Rolling OLS refit on the first grid day of each month over the trailing 10y (min 5y).
The residual on day t uses the most recent fit made on or before t. A refit day's own
close is in its training window (known at that close), so 1 day in ~21 is in-sample. F3 = e(t): negative = 10y rich versus fundamentals.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ust_rv.sources import resolve_source

INPUTS = {
    "y10": {"provider": "fred", "series": "DGS10"},
    "ff": {"provider": "fred", "series": "DFF"},
    "core_pce_12m": {"provider": "alfred", "series": "PCEPILFE", "transform": {"type": "pct_change", "periods": 12}},
    "real_gdp_yoy": {"provider": "alfred", "series": "GDPC1", "transform": {"type": "pct_change", "periods": 4}},
    "walcl": {"provider": "alfred", "series": "WALCL"},  # $mn
    "ngdp": {"provider": "alfred", "series": "GDP"},  # $bn SAAR
}
REGRESSORS = ["ff", "core_pce_12m", "real_gdp_yoy", "walcl_pct_gdp"]
R2_FLAG = 0.5


@dataclass
class FairValueResult:
    frame: pd.DataFrame  # value (residual, %), as_of, pit on the grid
    fits: pd.DataFrame  # per refit date: r2, nobs, const and coefficients
    fitted: pd.Series  # fitted 10y on the grid

    @property
    def latest_r2(self) -> float:
        return float(self.fits["r2"].iloc[-1]) if not self.fits.empty else float("nan")

    @property
    def flagged(self) -> bool:
        return not self.latest_r2 >= R2_FLAG


def rolling_fit(y: pd.Series, X: pd.DataFrame, window: int, min_periods: int) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Monthly-refit rolling OLS. Returns (residual, fitted, fits)."""
    data = pd.concat([y.rename("y"), X], axis=1)
    refit_dates = data.index.to_series().groupby(data.index.to_period("M")).first()
    rows = []
    for d in refit_dates:
        train = data.loc[:d].dropna().tail(window)
        if len(train) < min_periods:
            continue
        A = np.column_stack([np.ones(len(train)), train[X.columns].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, train["y"].to_numpy(), rcond=None)
        resid = train["y"].to_numpy() - A @ beta
        r2 = 1.0 - resid.var() / train["y"].to_numpy().var()
        rows.append({"date": d, "r2": r2, "nobs": len(train), "const": beta[0], **dict(zip(X.columns, beta[1:]))})
    fits = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["r2", "nobs", "const", *X.columns])
    if fits.empty:
        empty = pd.Series(np.nan, index=data.index)
        return empty, empty, fits
    coefs = fits[["const", *X.columns]].reindex(data.index, method="ffill")
    fitted = coefs["const"] + (coefs[X.columns] * X).sum(axis=1, min_count=len(X.columns))
    return data["y"] - fitted, fitted, fits


def build(providers, grid, window: int, min_periods: int) -> FairValueResult:
    parts = {k: resolve_source(spec, {"type": "level"}, providers, grid).frame for k, spec in INPUTS.items()}
    X = pd.DataFrame(
        {
            "ff": parts["ff"]["value"],
            "core_pce_12m": parts["core_pce_12m"]["value"],
            "real_gdp_yoy": parts["real_gdp_yoy"]["value"],
            "walcl_pct_gdp": parts["walcl"]["value"] / (parts["ngdp"]["value"] * 1000.0) * 100.0,
        }
    )
    resid, fitted, fits = rolling_fit(parts["y10"]["value"], X, window, min_periods)
    as_of = pd.concat([p["as_of"] for p in parts.values()], axis=1).min(axis=1, skipna=False)
    pit = pd.concat([p["pit"].fillna(False).astype(bool) for p in parts.values()], axis=1).all(axis=1)
    frame = pd.DataFrame({"value": resid, "as_of": as_of, "pit": pit.astype("boolean")})
    frame.loc[frame["value"].isna(), ["as_of", "pit"]] = pd.NA
    return FairValueResult(frame, fits, fitted)
