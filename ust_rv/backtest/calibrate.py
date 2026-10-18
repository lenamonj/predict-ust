"""Evidence-based calibration (Phase 3): nested walk-forward selection of model and decision
thresholds per tenor, live configuration written to settings.yaml, results memo written to
runs/phase3_results.md.

    python -m ust_rv.backtest.calibrate
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ust_rv.backtest import engine, metrics, selection
from ust_rv.config import CONFIG_DIR, repo_path, settings
from ust_rv.models.challenger_gbm import agreement
from ust_rv.models.regime_hmm import LABELS
from ust_rv.panel import build_panel
from ust_rv.scoring.scorecard import compute_scores

SPEC_PAIR = (0.50, 0.25)
MODELS = ("phase1", "ridge", "regime")


def write_settings(config: dict[str, tuple], path: Path = CONFIG_DIR / "settings.yaml") -> None:
    """config: tenor -> (model, enter, exit). Rewrites only the decision and composite_model blocks."""
    text = path.read_text(encoding="utf-8")
    decision = "decision:\n" + "".join(f"  {t}: {{enter: {e:.2f}, exit: {x:.2f}}}\n" for t, (_, e, x) in config.items())
    model = "composite_model:  # phase1 | ridge | regime\n" + "".join(f"  {t}: {m}\n" for t, (m, _, _) in config.items())
    # (?m) only: '.' must not cross lines, so each block ends at its last indented line
    text = re.sub(r"(?m)^decision:\n(?:  .*\n)+", decision, text)
    text = re.sub(r"(?m)^composite_model:.*\n(?:  .*\n)+", model, text)
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------- memo helpers ----------

def _f(x, fmt="{:.2f}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def _ci(ci, fmt="{:.2f}"):
    lo, hi = ci if isinstance(ci, tuple) else (np.nan, np.nan)
    return "-" if not np.isfinite(lo) else f"[{fmt.format(lo)}, {fmt.format(hi)}]"


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _key(k) -> str:
    m, e, x = k
    return f"{m} {e:.2f}/{x:.2f}"


def metrics_table(bt: engine.TenorBacktest) -> str:
    rows = []
    for (window, signal), m in bt.metrics.iterrows():
        rows.append({
            "window": window, "signal": signal,
            "IC": _f(m.get("ic")), "IC 95% CI": _ci(m.get("ic_ci")), "HAC t": _f(m.get("hac_t_hac")),
            "hit rate": _f(m["hit_rate"], "{:.0%}"), "hit 95% CI": _ci(m["hit_rate_ci"], "{:.0%}"),
            "Sharpe": _f(m["overlay_sharpe"]), "Sharpe 95% CI": _ci(m["sharpe_ci"]),
            "bp/yr": _f(m["pnl_bp_per_year"], "{:.0f}"), "max adverse bp": _f(m["max_adverse_bp"], "{:.0f}"),
            "neutral": _f(m["neutral_share"], "{:.0%}"), "avg hold d": _f(m["avg_hold_days"], "{:.0f}"),
            "turnover/yr": _f(m["turnover_per_year"], "{:.1f}"),
        })
    return _md(pd.DataFrame(rows))


def nested_table(n: selection.NestedResult) -> str:
    rows = []
    for (rule, tenor), m in n.summary.iterrows():
        rows.append({"rule": rule, "description": selection.RULES[rule], "tenor": tenor,
                     "nested Sharpe": _f(m["sharpe"]), "Sharpe 95% CI": _ci(m["sharpe_ci"]), "bp/yr": _f(m["bp_per_year"], "{:.0f}"),
                     "hit rate": _f(m["hit_rate"], "{:.0%}"), "neutral": _f(m["neutral_share"], "{:.0%}"),
                     "avg hold d": _f(m["avg_hold_days"], "{:.0f}"), "turnover/yr": _f(m["turnover_per_year"], "{:.1f}")})
    return _md(pd.DataFrame(rows))


def picks_table(n: selection.NestedResult, rule: str) -> str:
    p = n.picks.loc[rule].T.map(_key)
    return _md(p.reset_index().rename(columns={"year": "year"}))


def regime_section(regime) -> list[str]:
    f = regime.latest
    tm = pd.DataFrame(f.transmat, index=LABELS, columns=LABELS).round(3).reset_index().rename(columns={"index": "from \\ to"})
    means = pd.DataFrame(f.means, index=LABELS, columns=["rv21", "slope", "eq_bond_corr60", "credit_chg21"]).round(2).reset_index()
    probs = regime.probs.dropna()
    share = probs.idxmax(axis=1).value_counts(normalize=True).reindex(LABELS).fillna(0)
    return [
        f"Latest fit {max(regime.fits):%Y-%m-%d} ({len(regime.fits)} annual fits since {min(regime.fits):%Y}). Expected durations (days): "
        + ", ".join(f"{s} {d:.0f}" for s, d in zip(LABELS, f.expected_duration)) + ". Share of out-of-sample days by most likely state: "
        + ", ".join(f"{s} {v:.0%}" for s, v in share.items()) + f". Today: " + ", ".join(f"{s} {v:.2f}" for s, v in probs.iloc[-1].items()) + ".",
        "", "State means (feature z-scores):", "", _md(means.rename(columns={"index": "state"})), "", "Transition matrix:", "", _md(tm), "",
    ]


def challenger_section(results: dict[str, engine.TenorBacktest], config: dict[str, tuple]) -> list[str]:
    lines = []
    for tenor, bt in results.items():
        ch = bt.challenger
        model = config[tenor][0]
        agree = agreement(bt.signals["challenger"].loc[engine.OOS_START:], bt.signals[model].loc[engine.OOS_START:])
        today = "agree" if bt.signals["challenger"].iloc[-1] == bt.signals[model].iloc[-1] else "DISAGREE"
        m = bt.metrics.loc[("2009+", "challenger")]
        lines.append(
            f"- **{tenor}:** 2009+ IC {_f(m['ic'])}, Sharpe {_f(m['overlay_sharpe'])}; agreement with the live composite "
            f"{agree:.0%} of days, today {today}. Trees chosen by early stopping per refit: {', '.join(str(int(v)) for v in ch.n_trees)}."
        )
    lines.append(
        "- Early stopping often keeps one or a few trees: the validation fold finds no gain beyond the first splits, so the "
        "challenger has no demonstrated out-of-sample skill. Treat a DISAGREE badge as informational, not as a reason to override the composite."
    )
    return lines


def write_memo(path: Path, panel, regime, nested: selection.NestedResult, rule: str, config: dict, results) -> None:
    mean = nested.summary["sharpe"].astype(float).groupby(level="rule").mean()
    regime_on = any(m == "regime" for m, _, _ in config.values())
    lines = [
        "# Phase 3 results: regime model, challenger, evidence-based calibration",
        "",
        f"Generated by `python -m ust_rv.backtest.calibrate` on {pd.Timestamp.today():%Y-%m-%d}, data as of {panel.date:%Y-%m-%d}. "
        "Targets are overlapping 6m changes: ~30 independent observations per tenor in the 2012+ nested window. Read every interval as wide.",
        "",
        "## Method",
        "",
        "Nested walk-forward selection: each January from 2012, a rule picks (model, enter, exit) per tenor using only out-of-sample "
        "P&L and signal statistics realized since 2009 and before that January, then trades it for the year. Candidates: models "
        f"{', '.join(MODELS)} x 20 threshold pairs. Constraint: neutral share 40-55% and average hold > 63 days (least violation if none qualify). "
        f"The live rule is the simplest whose tenor-averaged nested Sharpe is within {selection.PARSIMONY_MARGIN} of the best. "
        f"The static reference uses the untouched spec pair {SPEC_PAIR[0]:.2f}/{SPEC_PAIR[1]:.2f}, so it carries no selection from this data.",
        "",
        "## Decision",
        "",
        f"- **Rule: `{rule}`** ({selection.RULES[rule]}). Tenor-averaged nested Sharpe: "
        + ", ".join(f"{r} {mean[r]:.2f}" for r in selection.RULES) + ".",
        "- **Live configuration written to settings.yaml:** " + "; ".join(f"{t} {_key(k)}" for t, k in config.items()) + ".",
        f"- **Regime weights: {'ON' if regime_on else 'OFF'}.** "
        + ("The selection rule picked the regime composite for at least one tenor." if regime_on else
           "Build_Plan 13 keeps them off unless they improve out-of-sample Sharpe. With selection done honestly (rule c can choose the "
           f"regime model each year), the result is worse than Phase 1 alone: rule c nested Sharpe 2y {_f(nested.summary.loc[('c_per_tenor_model_pair', '2y'), 'sharpe'])}, "
           f"10y {_f(nested.summary.loc[('c_per_tenor_model_pair', '10y'), 'sharpe'])}, vs static 2y {_f(nested.summary.loc[('static', '2y'), 'sharpe'])}, "
           f"10y {_f(nested.summary.loc[('static', '10y'), 'sharpe'])}. The regime composite looks better on 10y over 2009+ at fixed thresholds, "
           "but the edge does not survive being chosen in real time."),
        "- **10y:** no configuration earns a positive nested Sharpe with a CI excluding zero. The signal still reports -1/0/+1 each day "
        "under the live configuration; treat it as low conviction.",
        "",
        "## Nested out-of-sample results (2012+)",
        "",
        nested_table(nested),
        "",
        f"### Picks by year, rule `c_per_tenor_model_pair` (shows how often the regime model was chosen)",
        "",
        picks_table(nested, "c_per_tenor_model_pair"),
        "",
        "## Regime model",
        "",
        *regime_section(regime),
        "## Challenger (LightGBM, advisory)",
        "",
        *challenger_section(results, config),
        "",
        "## Metrics at the live configuration (all signals)",
        "",
        "Windows as in Phase 2. The ridge, regime and challenger rows start at the first 2009 refit. These are fixed-threshold "
        "walk-forward results, not nested; the nested table above is the selection-honest comparison.",
        "",
    ]
    for tenor, bt in results.items():
        lines += [f"### {tenor}", "", metrics_table(bt), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    panel = build_panel()
    scores = compute_scores(panel)
    regime = engine.regime_walk_forward(panel)

    spec = {t: SPEC_PAIR for t in engine.TENOR_SPEC}
    at_spec = engine.run(panel, scores, spec, n_draws=100, regime=regime)
    cands = {t: selection.build_candidates(bt, panel, MODELS) for t, bt in at_spec.items()}
    nested = selection.nested(cands, {t: bt.dy for t, bt in at_spec.items()}, SPEC_PAIR)
    rule = selection.best_rule(nested)
    config = selection.live_config(rule, cands, SPEC_PAIR, panel.date)

    write_settings(config)
    final = engine.run(panel, scores, {t: (e, x) for t, (_, e, x) in config.items()}, regime=regime)
    memo = repo_path("runs") / "phase3_results.md"
    write_memo(memo, panel, regime, nested, rule, config, final)
    print(f"rule={rule} config={config}\nmemo: {memo}")


if __name__ == "__main__":
    main()
