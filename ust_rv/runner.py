"""One run: refresh data, build the panel, score, persist a dated snapshot, diff vs the prior run."""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ust_rv.config import bloomberg_tickers
from ust_rv.config import factors as load_factors
from ust_rv.config import composite_model, repo_path, thresholds
from ust_rv.panel import Panel, build_panel
from ust_rv.scoring.decision import curve_read, next_threshold
from ust_rv.scoring.scorecard import TENORS, Scores, compute_scores, scorecard, theme_table

MOVER_DZ = 0.5


@dataclass
class RunResult:
    run_ts: pd.Timestamp
    panel: Panel
    scores: Scores
    card: pd.DataFrame
    themes: pd.DataFrame
    signal: dict  # composite, rv, next threshold per tenor, curve read
    prior: dict | None  # prior run's signal row
    movers: pd.DataFrame  # factors whose z moved > MOVER_DZ since the prior snapshot
    unverified_tickers: list[str]
    provider_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtests: dict | None = None  # tenor -> TenorBacktest (live model and benchmarks), when run(with_backtest=True)


def _unverified_in_use(lib: list[dict]) -> list[str]:
    weighted = {f["id"] for f in lib if f["weight_2y"] > 0 or f["weight_10y"] > 0}
    return [t for t, v in bloomberg_tickers().items() if not v["verified"] and weighted & set(v["used_by"])]


def use_learned_composites(panel: Panel, scores: Scores) -> None:
    """For tenors configured with a learned model (ridge or regime), replace the Phase 1 composite and RV
    with that model's walk-forward composite (latest refit weights)."""
    from ust_rv.backtest import engine  # heavy imports only when a learned model is selected

    learned = {t: composite_model(t) for t in TENORS if composite_model(t) != "phase1"}
    regime = engine.regime_walk_forward(panel) if "regime" in learned.values() else None
    for tenor, model in learned.items():
        bt = engine.tenor_backtest(panel, scores, tenor, *thresholds(tenor), regime=regime)
        scores.composite[tenor] = bt.composites[model]
        scores.rv[tenor] = bt.signals[model]


def _signal(scores: Scores) -> dict:
    out = {}
    for tenor in TENORS:
        comp = float(scores.composite[tenor].iloc[-1])
        rv = int(scores.rv[tenor].iloc[-1])
        label, dist = next_threshold(comp, rv, *thresholds(tenor))
        out[tenor] = {"composite": comp, "rv": rv, "next_threshold": label, "distance": dist, "model": composite_model(tenor)}
    out["curve"] = curve_read(out["2y"]["rv"], out["10y"]["rv"])
    return out


def _prior_snapshot(snap_dir: Path, today: pd.Timestamp) -> pd.DataFrame | None:
    files = sorted(p for p in snap_dir.glob("*.parquet") if pd.Timestamp(p.stem) < today)
    return pd.read_parquet(files[-1]) if files else None


def _prior_signal(history_path: Path, today: pd.Timestamp) -> dict | None:
    if not history_path.exists():
        return None
    h = pd.read_parquet(history_path)
    h = h[h["run_date"] < today]
    return h.iloc[-1].to_dict() if not h.empty else None


def run(providers: dict | None = None, factor_list: list[dict] | None = None, runs_dir: Path | None = None, today=None,
        with_backtest: bool = False) -> RunResult:
    """with_backtest: also run the walk-forward backtest of the live model and its benchmarks."""
    lib = factor_list if factor_list is not None else load_factors()
    today = pd.Timestamp(today).normalize() if today is not None else pd.Timestamp.today().normalize()
    runs_dir = runs_dir or repo_path("runs")
    snap_dir = runs_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    history_path = runs_dir / "signal_history.parquet"

    panel = build_panel(providers, lib, end=today)
    if factor_list is None:
        panel.save()
    scores = compute_scores(panel, lib)
    if any(composite_model(t) != "phase1" for t in TENORS):
        use_learned_composites(panel, scores)
    card = scorecard(panel, scores, lib)
    signal = _signal(scores)

    prior_card = _prior_snapshot(snap_dir, today)
    prior = _prior_signal(history_path, today)
    if prior_card is not None:
        dz = (card["z"] - prior_card["z"].reindex(card.index)).rename("dz")
        movers = pd.concat([card[["factor", "theme", "z"]], dz], axis=1)
        movers = movers[movers["dz"].abs() > MOVER_DZ].sort_values("dz", key=abs, ascending=False)
    else:
        movers = pd.DataFrame(columns=["factor", "theme", "z", "dz"])

    card.to_parquet(snap_dir / f"{today:%Y-%m-%d}.parquet")
    row = pd.DataFrame(
        [
            {
                "run_ts": pd.Timestamp.now(),
                "run_date": today,
                "panel_date": panel.date,
                "market_date": panel.market_date,
                "composite_2y": signal["2y"]["composite"],
                "composite_10y": signal["10y"]["composite"],
                "rv_2y": signal["2y"]["rv"],
                "rv_10y": signal["10y"]["rv"],
                "curve": signal["curve"],
            }
        ]
    )
    history = pd.concat([pd.read_parquet(history_path), row], ignore_index=True) if history_path.exists() else row
    history.to_parquet(history_path, index=False)

    log = pd.DataFrame(
        [{"id": k, "status": m["status"], "source": m["source"], "label": m["label"], "stale": m["stale"], "errors": "; ".join(m["errors"])} for k, m in panel.meta.items()]
    ).set_index("id")
    backtests = None
    if with_backtest:
        from ust_rv.backtest import engine

        backtests = engine.run(panel, scores, {t: thresholds(t) for t in TENORS}, n_draws=200)
    return RunResult(pd.Timestamp.now(), panel, scores, card, theme_table(scores), signal, prior, movers,
                     _unverified_in_use(lib), log, backtests)
