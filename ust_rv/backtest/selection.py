"""Evidence-based calibration by nested walk-forward selection.

Each January, a selection rule picks a configuration (model, enter, exit) using only P&L and
signal statistics realized before that January, and trades it for the year. Chaining the years
gives a selection-honest out-of-sample record for each rule. P&L is realized daily, so no
target lag is needed for selection."""
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from ust_rv.backtest import engine, metrics
from ust_rv.backtest.walkforward import january_refit_dates
from ust_rv.scoring.decision import hysteresis

FIRST_SELECTION_YEAR = 2012  # three years of out-of-sample history since 2009
PAIRS = [(e, x) for e, x in product(engine.ENTER_GRID, engine.EXIT_GRID) if x <= e]


@dataclass
class Candidate:
    model: str
    enter: float
    exit: float

    @property
    def key(self) -> tuple:
        return (self.model, self.enter, self.exit)


@dataclass
class Candidates:
    """rv and daily P&L for every (model, pair) on one tenor, out of sample since OOS_START."""
    rv: pd.DataFrame  # columns: MultiIndex (model, enter, exit)
    pnl: pd.DataFrame


def build_candidates(bt: engine.TenorBacktest, panel, models: tuple[str, ...]) -> Candidates:
    ycol, scol, s_years, years, _ = engine.TENOR_SPEC[bt.tenor]
    y = panel.yields.loc[engine.OOS_START:]
    rv, pnl = {}, {}
    for model in models:
        comp = bt.composites[model].loc[engine.OOS_START:]
        for enter, exit_ in PAIRS:
            r = hysteresis(comp, enter, exit_).fillna(0.0)
            rv[(model, enter, exit_)] = r
            pnl[(model, enter, exit_)] = metrics.overlay_pnl(r, y[ycol], y[scol], s_years, years, y["ff"])
    return Candidates(pd.DataFrame(rv), pd.DataFrame(pnl))


def _stats(c: Candidates, end: pd.Timestamp) -> pd.DataFrame:
    rv, pnl = c.rv.loc[: end - pd.Timedelta(days=1)], c.pnl.loc[: end - pd.Timedelta(days=1)]
    m = pnl.resample("ME").sum()
    out = pd.DataFrame({"sharpe": m.mean() / m.std().replace(0, np.nan) * np.sqrt(12)})
    out["neutral_share"] = (rv == 0).mean()
    out["avg_hold_days"] = pd.Series({k: metrics.avg_hold(rv[k]) for k in rv.columns})
    lo, hi = engine.NEUTRAL_RANGE
    out["viol"] = (lo - out["neutral_share"]).clip(lower=0) + (out["neutral_share"] - hi).clip(lower=0) \
        + (engine.MIN_HOLD_DAYS - out["avg_hold_days"]).clip(lower=0) / engine.MIN_HOLD_DAYS
    return out


def select(stats: pd.DataFrame, allowed=None) -> tuple:
    """Best Sharpe among candidates meeting the constraints; else least violation, then best Sharpe."""
    s = stats if allowed is None else stats.loc[[k for k in stats.index if allowed(k)]]
    ok = s[s["viol"] == 0]
    if not ok.empty:
        return ok["sharpe"].idxmax()
    return s.sort_values(["viol", "sharpe"], ascending=[True, False]).index[0]


def select_shared(stats_by_tenor: dict[str, pd.DataFrame], model: str) -> tuple:
    """One pair for both tenors: qualifies only if it qualifies on every tenor; ranked by mean Sharpe."""
    joined = sum(s.loc[model][["viol"]] for s in stats_by_tenor.values())
    joined["sharpe"] = sum(s.loc[model]["sharpe"] for s in stats_by_tenor.values()) / len(stats_by_tenor)
    pair = select(joined)
    return (model, *pair)


RULES = {
    "static": "Phase 1 at the spec pair (0.50/0.25), no selection",
    "a_shared": "Shared pair for both tenors, Phase 1",
    "b_per_tenor_pair": "Per-tenor pair, Phase 1",
    "c_per_tenor_model_pair": "Per-tenor model and pair",
}


def choose(rule: str, stats_by_tenor: dict[str, pd.DataFrame], static_pair: tuple[float, float]) -> dict[str, tuple]:
    if rule == "static":
        return {t: ("phase1", *static_pair) for t in stats_by_tenor}
    if rule == "a_shared":
        key = select_shared(stats_by_tenor, "phase1")
        return {t: key for t in stats_by_tenor}
    if rule == "b_per_tenor_pair":
        return {t: select(s, allowed=lambda k: k[0] == "phase1") for t, s in stats_by_tenor.items()}
    if rule == "c_per_tenor_model_pair":
        return {t: select(s) for t, s in stats_by_tenor.items()}
    raise ValueError(rule)


@dataclass
class NestedResult:
    picks: pd.DataFrame  # (rule, tenor) x year -> chosen key
    pnl: dict  # (rule, tenor) -> nested out-of-sample daily P&L
    rv: dict
    summary: pd.DataFrame  # (rule, tenor) -> nested metrics


def nested(cands: dict[str, Candidates], dy: dict[str, pd.Series], static_pair: tuple[float, float]) -> NestedResult:
    index = next(iter(cands.values())).rv.index
    years = [d for d in january_refit_dates(index) if d.year >= FIRST_SELECTION_YEAR]
    bounds = years + [index[-1] + pd.Timedelta(days=1)]
    picks, pnl, rv = {}, {}, {}
    for rule in RULES:
        series = {t: ([], []) for t in cands}
        for k, start in enumerate(years):
            stats = {t: _stats(c, start) for t, c in cands.items()}
            chosen = choose(rule, stats, static_pair)
            for t, key in chosen.items():
                picks[(rule, t, start.year)] = key
                period = (index >= start) & (index < bounds[k + 1])
                series[t][0].append(cands[t].pnl.loc[period, key])
                series[t][1].append(cands[t].rv.loc[period, key])
        for t, (p, r) in series.items():
            pnl[(rule, t)], rv[(rule, t)] = pd.concat(p), pd.concat(r)
    rows = {}
    for (rule, t), p in pnl.items():
        r = rv[(rule, t)]
        rows[(rule, t)] = {
            "sharpe": metrics.sharpe(p),
            "sharpe_ci": metrics.block_bootstrap(metrics.monthly(p).to_frame("pnl"),
                                                 lambda f: metrics._annualized_sharpe(f["pnl"]), block=6, n_draws=500),
            "bp_per_year": float(p.sum() / ((p.index[-1] - p.index[0]).days / 365.25)),
            "hit_rate": metrics.hit_rate(r, dy[t].reindex(r.index)),
            "neutral_share": metrics.neutral_share(r),
            "avg_hold_days": metrics.avg_hold(r),
            "turnover_per_year": metrics.turnover(r),
        }
    picks_df = pd.Series(picks).rename_axis(["rule", "tenor", "year"]).unstack("year")
    summary = pd.DataFrame(rows).T.rename_axis(["rule", "tenor"])
    return NestedResult(picks_df, pnl, rv, summary)


PARSIMONY_MARGIN = 0.05  # a more flexible rule must beat a simpler one by this much nested Sharpe


def best_rule(result: NestedResult) -> str:
    """Simplest rule (RULES order) whose tenor-averaged nested Sharpe is within the margin of the best."""
    mean = result.summary["sharpe"].astype(float).groupby(level="rule").mean()
    return next(r for r in RULES if mean[r] >= mean.max() - PARSIMONY_MARGIN)


def live_config(rule: str, cands: dict[str, Candidates], static_pair: tuple[float, float], asof: pd.Timestamp) -> dict[str, tuple]:
    """The winning rule applied to all out-of-sample history through `asof`."""
    stats = {t: _stats(c, asof + pd.Timedelta(days=1)) for t, c in cands.items()}
    return {k: (m, float(e), float(x)) for k, (m, e, x) in choose(rule, stats, static_pair).items()}
