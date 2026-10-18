"""Phase 2 backtest: walk-forward ridge, Phase 1 composite and baselines, scored per tenor
over several windows, plus the decision-threshold sweep."""
from dataclasses import dataclass, field
from itertools import product

import pandas as pd

from ust_rv.backtest import baselines, metrics, walkforward
from ust_rv.config import factors as load_factors
from ust_rv.config import repo_path, settings
from ust_rv.features.zscore import rolling_z
from ust_rv.panel import Panel
from ust_rv.scoring.decision import hysteresis
from ust_rv.scoring.scorecard import Scores

# tenor -> (yield column, roll point column, roll point maturity, maturity, momentum factor)
TENOR_SPEC = {"2y": ("y2", "y1", 1.0, 2.0, "G2"), "10y": ("y10", "y7", 7.0, 10.0, "G1")}
OOS_START = "2009-01-01"
ENTER_GRID = (0.3, 0.4, 0.5, 0.6, 0.75)
EXIT_GRID = (0.1, 0.2, 0.25, 0.3)
NEUTRAL_RANGE = (0.40, 0.55)
MIN_HOLD_DAYS = 63
MIN_TRAIN_DAYS = 8 * 252
EMBARGO_DAYS = 21
PIT_SHARE_MIN = 0.90


@dataclass
class TenorBacktest:
    tenor: str
    composites: pd.DataFrame  # phase1, ridge, regime (when a regime model is supplied)
    signals: pd.DataFrame  # rv per signal name
    pnl: pd.DataFrame  # daily overlay P&L per signal name
    dy: pd.Series
    walkforward: walkforward.WalkForward
    pit_share: pd.Series
    regime: walkforward.RegimeComposite | None = None
    challenger: object | None = None  # ChallengerResult (advisory)
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)  # (window, signal) x metric


def pit_share(panel: Panel, tenor: str) -> pd.Series:
    """Share of weighted, available factors whose value on each date is true point-in-time."""
    ids = [f["id"] for f in load_factors() if f[f"weight_{tenor}"] > 0 and panel.meta[f["id"]]["status"] in ("ok", "stale")]
    pit = panel.pit[ids].astype("boolean")
    return pit.fillna(False).astype(bool).sum(axis=1) / pit.notna().sum(axis=1).where(lambda n: n > 0)


def windows(pit: pd.Series) -> dict[str, pd.DatetimeIndex]:
    idx = pit.index
    return {
        "2005+": idx[idx >= pd.Timestamp(settings()["backtest_start"])],
        "2009+": idx[idx >= pd.Timestamp(OOS_START)],
        "PIT>=90%": idx[(idx >= pd.Timestamp(OOS_START)) & (pit >= PIT_SHARE_MIN).to_numpy()],
    }


LEARNED_MODELS = ("ridge", "regime", "challenger")  # nothing before the first 2009 refit


def regime_walk_forward(panel: Panel):
    from ust_rv.models import regime_hmm

    zc = settings()["zscore"]
    feat = regime_hmm.features(panel.yields, zc["window"], zc["min_periods"], zc["clip"])
    return regime_hmm.walk_forward(feat, cache_dir=repo_path("cache") / "models")


def tenor_backtest(panel: Panel, scores: Scores, tenor: str, enter: float, exit_: float, regime=None) -> TenorBacktest:
    cfg = settings()
    ycol, scol, s_years, years, mom_id = TENOR_SPEC[tenor]
    y, yields = panel.yields[ycol], panel.yields
    dy = scores.dy_h[tenor]
    target = dy / metrics.realized_vol_6m(y, cfg["horizon_days"])

    zc = cfg["zscore"]
    std = lambda c: rolling_z(c, zc["window"], zc["min_periods"], zc["clip"])  # noqa: E731
    wf = walkforward.run(scores.theme_z[tenor], target, cfg["horizon_days"], EMBARGO_DAYS, MIN_TRAIN_DAYS, standardize=std)
    comps = pd.DataFrame({"phase1": scores.composite[tenor], "ridge": wf.composite})
    rc = ch = None
    if regime is not None:
        rc = walkforward.run_regime(scores.theme_z[tenor], target, wf, regime, std, cfg["horizon_days"], EMBARGO_DAYS)
        comps["regime"] = rc.composite
        ch = _challenger(panel, scores, tenor, regime, target, wf.weights.index, enter, exit_)
        comps["challenger"] = ch.pred_z
    idx = comps.index
    signals = pd.DataFrame(
        {
            **{m: hysteresis(comps[m], enter, exit_) for m in comps.columns},
            "always_long": baselines.always_long(idx),
            "momentum": baselines.momentum(panel.values[mom_id]),
            "random_walk": baselines.random_walk(idx),
        }
    )
    pnl = signals.apply(lambda rv: metrics.overlay_pnl(rv, y, yields[scol], s_years, years, yields["ff"]))
    return TenorBacktest(tenor, comps, signals, pnl, dy, wf, pit_share(panel, tenor), rc, ch)


def _challenger(panel: Panel, scores: Scores, tenor: str, regime, target, refits, enter: float, exit_: float):
    from ust_rv.models import challenger_gbm

    cfg = settings()
    available = {k for k, m in panel.meta.items() if m["status"] in ("ok", "stale")}
    tw = pd.Series({k: v[f"w{tenor}"] for k, v in cfg["theme_weights"].items()})
    ids = challenger_gbm.top_factors(load_factors(), tw, tenor, available)
    return challenger_gbm.run(scores.theme_z[tenor], scores.pressure, ids, regime, target, refits,
                              cfg["horizon_days"], EMBARGO_DAYS, enter, exit_)


def score(bt: TenorBacktest, n_draws: int = 500) -> pd.DataFrame:
    """Section 11.2 metrics for every signal over every window."""
    rows = {}
    for wname, widx in windows(bt.pit_share).items():
        for sig in bt.signals.columns:
            if wname == "2005+" and sig in LEARNED_MODELS:
                continue
            rv = bt.signals[sig].reindex(widx).dropna()
            if rv.empty:
                continue
            comp = bt.composites[sig].reindex(widx) if sig in bt.composites else None
            rows[(wname, sig)] = metrics.summarize(rv, bt.dy, bt.pnl[sig], comp, n_draws=n_draws)
    out = pd.DataFrame(rows).T
    out.index.names = ["window", "signal"]
    return out


def sweep(composite: pd.Series, dy: pd.Series, pnl_fn, start: str = OOS_START) -> pd.DataFrame:
    """Threshold grid (Build_Plan Section 10) over one composite. pnl_fn(rv) -> daily P&L."""
    comp = composite.loc[start:]
    ic = metrics.ic(comp, dy)
    rows = []
    for enter, exit_ in product(ENTER_GRID, EXIT_GRID):
        if exit_ > enter:
            continue
        rv = hysteresis(comp, enter, exit_).dropna()
        rows.append(
            {
                "enter": enter,
                "exit": exit_,
                "neutral_share": metrics.neutral_share(rv),
                "avg_hold_days": metrics.avg_hold(rv),
                "hit_rate": metrics.hit_rate(rv, dy.reindex(rv.index)),
                "ic": ic,
                "overlay_sharpe": metrics.sharpe(pnl_fn(rv)),
                "turnover_per_year": metrics.turnover(rv),
            }
        )
    return pd.DataFrame(rows)


def select_thresholds(sweeps: dict[str, pd.DataFrame]) -> tuple[float, float, bool]:
    """Pairs qualifying on every tenor (neutral share in range, average hold > 63d); best mean Sharpe.
    If none qualify, the pair with the smallest total constraint violation. Returns (enter, exit, qualified)."""
    keys = ["enter", "exit"]
    joined = None
    for tenor, s in sweeps.items():
        lo, hi = NEUTRAL_RANGE
        viol = (lo - s["neutral_share"]).clip(lower=0) + (s["neutral_share"] - hi).clip(lower=0) \
            + ((MIN_HOLD_DAYS - s["avg_hold_days"]).clip(lower=0) / MIN_HOLD_DAYS)
        part = s[keys].assign(**{f"sharpe_{tenor}": s["overlay_sharpe"], f"viol_{tenor}": viol})
        joined = part if joined is None else joined.merge(part, on=keys)
    joined["sharpe"] = joined.filter(like="sharpe_").mean(axis=1)
    joined["viol"] = joined.filter(like="viol_").sum(axis=1)
    ok = joined[joined["viol"] == 0]
    if not ok.empty:
        best = ok.sort_values("sharpe", ascending=False).iloc[0]
        return float(best["enter"]), float(best["exit"]), True
    best = joined.sort_values(["viol", "sharpe"], ascending=[True, False]).iloc[0]
    return float(best["enter"]), float(best["exit"]), False


def run(panel: Panel, scores: Scores, thresholds: dict[str, tuple[float, float]], n_draws: int = 500, regime=None) -> dict[str, TenorBacktest]:
    """thresholds: tenor -> (enter, exit). regime: RegimeWalkForward, or None to skip the regime composite."""
    out = {}
    for tenor in TENOR_SPEC:
        enter, exit_ = thresholds[tenor]
        bt = tenor_backtest(panel, scores, tenor, enter, exit_, regime)
        bt.metrics = score(bt, n_draws)
        out[tenor] = bt
    return out


def sweeps_for(results: dict[str, TenorBacktest], models: dict[str, str], panel: Panel) -> dict[str, pd.DataFrame]:
    out = {}
    for tenor, bt in results.items():
        ycol, scol, s_years, years, _ = TENOR_SPEC[tenor]
        y = panel.yields

        def pnl_fn(rv, y=y, ycol=ycol, scol=scol, s_years=s_years, years=years):
            return metrics.overlay_pnl(rv, y[ycol].reindex(rv.index), y[scol].reindex(rv.index), s_years, years, y["ff"].reindex(rv.index))

        out[tenor] = sweep(bt.composites[models[tenor]], bt.dy, pnl_fn)
    return out
