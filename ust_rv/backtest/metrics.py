"""Evaluation metrics (Build_Plan 11.2). Yields are in percent, yield changes in bp.
Composite pressure is in yield space (+ = yields higher); RV is in price space (+1 = long)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS = 252


# ---------- target scaling ----------

def realized_vol_6m(y: pd.Series, horizon: int = 126) -> pd.Series:
    """Trailing std of daily changes (bp) over `horizon` days, scaled to the horizon."""
    return (y.diff() * 100.0).rolling(horizon).std() * np.sqrt(horizon)


# ---------- signal statistics ----------

def ic(composite: pd.Series, dy: pd.Series) -> float:
    pairs = pd.concat([composite, dy], axis=1, sort=True).dropna()
    return float(pairs.iloc[:, 0].corr(pairs.iloc[:, 1], method="spearman")) if len(pairs) > 2 else float("nan")


def rolling_ic(composite: pd.Series, dy: pd.Series, window: int = 3 * TRADING_DAYS) -> pd.Series:
    """Spearman IC over the trailing `window` days, sampled at month-ends."""
    pairs = pd.concat([composite.rename("c"), dy.rename("d")], axis=1, sort=True).dropna()
    ends = pairs.index.to_series().groupby(pairs.index.to_period("M")).last()
    out = {e: ic(pairs.loc[:e, "c"].tail(window), pairs.loc[:e, "d"].tail(window)) for e in ends if pairs.loc[:e].shape[0] >= window}
    return pd.Series(out, dtype=float)


def hit_rate(rv: pd.Series, dy: pd.Series) -> float:
    """Share of non-zero signals where sign(RV) == sign(-dy). Zero yield changes are excluded."""
    pairs = pd.concat([rv, dy], axis=1, sort=True).dropna()
    pairs = pairs[(pairs.iloc[:, 0] != 0) & (pairs.iloc[:, 1] != 0)]
    if pairs.empty:
        return float("nan")
    return float((np.sign(pairs.iloc[:, 0]) == np.sign(-pairs.iloc[:, 1])).mean())


def neutral_share(rv: pd.Series) -> float:
    rv = rv.dropna()
    return float((rv == 0).mean()) if len(rv) else float("nan")


def _runs(rv: pd.Series) -> pd.DataFrame:
    rv = rv.dropna()
    run_id = (rv != rv.shift()).cumsum()
    return rv.groupby(run_id).agg(["first", "size"])


def avg_hold(rv: pd.Series) -> float:
    """Average length in grid days of consecutive identical non-zero states."""
    runs = _runs(rv)
    active = runs[runs["first"] != 0]
    return float(active["size"].mean()) if not active.empty else 0.0


def turnover(rv: pd.Series) -> float:
    """Signal changes per year."""
    rv = rv.dropna()
    if len(rv) < 2:
        return float("nan")
    years = (rv.index[-1] - rv.index[0]).days / 365.25
    return float((rv.diff().fillna(0) != 0).sum() / years)


def max_adverse(rv: pd.Series, dy: pd.Series) -> float:
    """Largest 6m yield move against an active signal, bp (a long loses when yields rise)."""
    against = (rv * dy)[rv != 0].dropna()
    return float(against.max()) if not against.empty else float("nan")


def calibration(composite: pd.Series, dy: pd.Series, bins: int = 10) -> pd.DataFrame:
    pairs = pd.concat([composite.rename("composite"), dy.rename("dy")], axis=1, sort=True).dropna()
    pairs["decile"] = pd.qcut(pairs["composite"], bins, labels=False, duplicates="drop") + 1
    return pairs.groupby("decile").agg(composite=("composite", "mean"), mean_dy_bp=("dy", "mean"), n=("dy", "size"))


# ---------- overlay P&L ----------

def par_duration(y_pct: pd.Series, years: float) -> pd.Series:
    """Modified duration of a semiannual par bond."""
    y = y_pct / 100.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (1.0 - (1.0 + y / 2.0) ** (-2.0 * years)) / y
    return d.where(y.abs() > 1e-6, years)


def pnl_components(rv: pd.Series, y: pd.Series, y_short: pd.Series, short_years: float, years: float, fin: pd.Series) -> pd.DataFrame:
    """Daily overlay P&L in bp of 1 unit DV01, split into direction, carry and roll. Signal at close t earns from t+1.
    Carry = (y - financing) / duration; roll = slope to the shorter point, both accrued per calendar day.
    A day with any component undefined earns nothing."""
    pos = rv.shift(1).fillna(0.0)
    days = y.index.to_series().diff().dt.days.fillna(0).to_numpy()
    dy_bp = y.diff() * 100.0
    carry = (y - fin).shift(1) * 100.0 / par_duration(y, years).shift(1) / 360.0 * days
    roll = ((y - y_short) * 100.0 / (years - short_years)).shift(1) / 365.0 * days
    out = pd.DataFrame({"direction": pos * -dy_bp, "carry": pos * carry, "roll": pos * roll})
    return out.where(out.notna().all(axis=1), 0.0)


def overlay_pnl(rv: pd.Series, y: pd.Series, y_short: pd.Series, short_years: float, years: float, fin: pd.Series) -> pd.Series:
    """Daily P&L in bp of 1 unit DV01 (sum of pnl_components)."""
    return pnl_components(rv, y, y_short, short_years, years, fin).sum(axis=1)


def drawdown(pnl: pd.Series) -> pd.Series:
    """Cumulative P&L minus its running peak (bp, <= 0)."""
    cum = pnl.cumsum()
    return cum - cum.cummax()


def monthly(pnl: pd.Series) -> pd.Series:
    return pnl.resample("ME").sum()


def _annualized_sharpe(monthly_pnl: pd.Series) -> float:
    sd = monthly_pnl.std()
    return float(monthly_pnl.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan")


def sharpe(pnl: pd.Series) -> float:
    return _annualized_sharpe(monthly(pnl))


# ---------- inference ----------

def block_bootstrap(frame: pd.DataFrame, stat, block: int, n_draws: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Moving block bootstrap CI of stat(frame) over rows."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    if n < 2 * block:
        return float("nan"), float("nan")
    n_blocks = int(np.ceil(n / block))
    values = frame.to_numpy()
    draws = []
    for _ in range(n_draws):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        draws.append(stat(pd.DataFrame(values[idx], columns=frame.columns)))
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def hac_regression(composite: pd.Series, dy: pd.Series, lags: int = 126) -> dict:
    """dy_h = a + b * composite with Newey-West standard errors (overlapping targets)."""
    pairs = pd.concat([composite.rename("x"), dy.rename("y")], axis=1, sort=True).dropna()
    X = sm.add_constant(pairs["x"])
    ols = sm.OLS(pairs["y"], X).fit()
    hac = sm.OLS(pairs["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"beta": float(hac.params["x"]), "t_hac": float(hac.tvalues["x"]), "t_ols": float(ols.tvalues["x"]), "n": int(len(pairs))}


# ---------- summary ----------

def summarize(rv: pd.Series, dy: pd.Series, pnl: pd.Series, composite: pd.Series | None = None, n_draws: int = 1000, seed: int = 0) -> dict:
    """All Section 11.2 metrics for one signal over the dates in `rv`."""
    rv = rv.dropna()
    dy = dy.reindex(rv.index)
    pnl = pnl.reindex(rv.index).fillna(0.0)
    out = {
        "hit_rate": hit_rate(rv, dy),
        "overlay_sharpe": sharpe(pnl),
        "pnl_bp_per_year": float(pnl.sum() / ((rv.index[-1] - rv.index[0]).days / 365.25)),
        "max_adverse_bp": max_adverse(rv, dy),
        "turnover_per_year": turnover(rv),
        "neutral_share": neutral_share(rv),
        "avg_hold_days": avg_hold(rv),
    }
    nan_ci = (float("nan"), float("nan"))
    active = bool((rv != 0).any())  # a never-active signal (random walk) has no hit rate or Sharpe to bound
    hr_frame = pd.concat([rv.rename("rv"), dy.rename("dy")], axis=1, sort=True).dropna()
    out["hit_rate_ci"] = block_bootstrap(hr_frame, lambda f: hit_rate(f["rv"], f["dy"]), block=126, n_draws=n_draws, seed=seed) if active else nan_ci
    m = monthly(pnl).to_frame("pnl")
    out["sharpe_ci"] = block_bootstrap(m, lambda f: _annualized_sharpe(f["pnl"]), block=6, n_draws=n_draws, seed=seed) if active else nan_ci
    if composite is not None:
        c = composite.reindex(rv.index)
        out["ic"] = ic(c, dy)
        ic_frame = pd.concat([c.rename("c"), dy.rename("dy")], axis=1, sort=True).dropna()
        out["ic_ci"] = block_bootstrap(ic_frame, lambda f: ic(f["c"], f["dy"]), block=126, n_draws=n_draws, seed=seed)
        out.update({f"hac_{k}": v for k, v in hac_regression(c, dy).items()})
    return out
