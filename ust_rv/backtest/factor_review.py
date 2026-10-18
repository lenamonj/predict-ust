"""Factor-level evidence: per-factor track records, leave-one-out effects, and a selection-honest
test of pruning factors in real time."""
import numpy as np
import pandas as pd

from ust_rv.backtest import metrics
from ust_rv.backtest.robustness import composite_from_pressure
from ust_rv.backtest.walkforward import january_refit_dates
from ust_rv.scoring.decision import hysteresis

ERAS = {"2005-08": ("2005-01-01", "2008-12-31"), "2009-15": ("2009-01-01", "2015-12-31"),
        "2016-19": ("2016-01-01", "2019-12-31"), "2020+": ("2020-01-01", None)}
LOO_START = "2009-01-01"


def _ic(x: pd.Series, dy: pd.Series, start, end=None) -> float:
    return metrics.ic(x.loc[start:end], dy)


def factor_table(z: pd.DataFrame, lib: pd.DataFrame, theme_weights: pd.Series, dy: pd.Series, tenor: str,
                 pit_dates: pd.DatetimeIndex, available: set[str], zc: dict) -> pd.DataFrame:
    """One row per weighted, available factor for this tenor."""
    w = lib[f"weight_{tenor}"].astype(float)
    ids = [f for f in lib.index if w[f] > 0 and f in available]
    weights = w.where(w.index.isin(ids), 0.0)
    pressure = z.mul(lib["sign"], axis=1)
    base = composite_from_pressure(pressure, weights, lib["theme"], theme_weights, zc)
    base_ic = _ic(base, dy, LOO_START)
    corr = pressure[ids].loc["2009":].corr()
    rows = []
    for f in ids:
        p = pressure[f]
        loo = composite_from_pressure(pressure, weights.drop(f).reindex(weights.index, fill_value=0.0), lib["theme"], theme_weights, zc)
        others = corr[f].drop(f)
        twin = others.abs().idxmax()
        recent_z = z[f].loc[z.index[-1] - pd.DateOffset(years=5):].dropna()
        rows.append({
            "id": f, "factor": lib.loc[f, "name"], "theme": lib.loc[f, "theme"], "sign": int(lib.loc[f, "sign"]), "weight": w[f],
            "ic_2005": _ic(p, dy, "2005-01-01"),
            **{f"ic_{k}": _ic(p, dy, a, b) for k, (a, b) in ERAS.items()},
            "ic_pit90": metrics.ic(p.reindex(pit_dates), dy),
            "eras_negative": sum(_ic(p, dy, a, b) < 0 for a, b in ERAS.values()),
            "loo_delta_ic": _ic(loo, dy, LOO_START) - base_ic,  # > 0: the composite is better without this factor
            "closest": twin, "closest_corr": float(others[twin]),
            "z_same_sign_5y": float(abs(np.sign(recent_z).mean())) if len(recent_z) else np.nan,  # 1 = stuck on one side
        })
    return pd.DataFrame(rows).set_index("id")


def theme_leave_one_out(z: pd.DataFrame, lib: pd.DataFrame, theme_weights: pd.Series, dy: pd.Series, tenor: str,
                        available: set[str], zc: dict) -> pd.Series:
    """Composite IC (2009+) with each theme removed, minus the full composite's IC."""
    w = lib[f"weight_{tenor}"].astype(float).where(lambda s: s.index.isin(list(available)), 0.0)
    pressure = z.mul(lib["sign"], axis=1)
    base_ic = _ic(composite_from_pressure(pressure, w, lib["theme"], theme_weights, zc), dy, LOO_START)
    out = {}
    for theme in theme_weights.index:
        w_t = w.where(lib["theme"] != theme, 0.0)
        tw = theme_weights.drop(theme)
        out[theme] = _ic(composite_from_pressure(pressure, w_t, lib["theme"], tw, zc), dy, LOO_START) - base_ic
    return pd.Series(out, name="delta_ic_without_theme")


# ---------- selection-honest pruning ----------

PRUNING_RULES = {
    "static": dict(drop_below=None, flip_below=None),
    "drop_ic_below_0": dict(drop_below=0.0, flip_below=None),
    "drop_ic_below_-0.10": dict(drop_below=-0.10, flip_below=None),
    "drop_-0.10_flip_-0.20": dict(drop_below=-0.10, flip_below=-0.20),
}
MIN_PAIRS = 3 * 252  # a factor needs 3 years of realized pairs before a rule may act on it


def nested_pruning(z: pd.DataFrame, lib: pd.DataFrame, theme_weights: pd.Series, dy: pd.Series, tenor: str, available: set[str],
                   zc: dict, horizon: int, rule: dict, first_year: int = 2012, ic_start: str = "2000-01-01") -> tuple[pd.Series, dict]:
    """Each January, judge every weighted factor on its IC over realized pairs only (t <= January minus the horizon),
    drop or flip per the rule, rebuild the composite with the survivors and keep that year's slice."""
    w0 = lib[f"weight_{tenor}"].astype(float).where(lambda s: s.index.isin(list(available)), 0.0)
    sign0 = lib["sign"].astype(float)
    idx = z.index
    years = [d for d in january_refit_dates(idx) if d.year >= first_year]
    bounds = years + [idx[-1] + pd.Timedelta(days=1)]
    comp = pd.Series(np.nan, index=idx)
    decisions = {}
    for k, start in enumerate(years):
        cutoff = idx[idx.get_loc(start) - horizon]
        w, sign = w0.copy(), sign0.copy()
        dropped, flipped = [], []
        for f in w0.index[w0 > 0]:
            p = (z[f] * sign0[f]).loc[ic_start:cutoff]
            pairs = pd.concat([p, dy.loc[:cutoff]], axis=1, sort=True).dropna()
            if len(pairs) < MIN_PAIRS:
                continue
            ic = metrics.ic(pairs.iloc[:, 0], pairs.iloc[:, 1])
            if rule["flip_below"] is not None and ic < rule["flip_below"]:
                sign[f] = -sign[f]
                flipped.append(f)
            elif rule["drop_below"] is not None and ic < rule["drop_below"]:
                w[f] = 0.0
                dropped.append(f)
        period = (idx >= start) & (idx < bounds[k + 1])
        comp[period] = composite_from_pressure(z.mul(sign, axis=1), w, lib["theme"], theme_weights, zc)[period]
        decisions[start.year] = {"dropped": dropped, "flipped": flipped}
    return comp, decisions


def score_rule(comp: pd.Series, dy: pd.Series, y: pd.DataFrame, tenor_spec: tuple, enter: float, exit_: float, start: str) -> dict:
    ycol, scol, s_years, years, _ = tenor_spec
    c = comp.loc[start:]
    rv = hysteresis(c, enter, exit_)
    yy = y.loc[start:]
    pnl = metrics.overlay_pnl(rv.fillna(0.0), yy[ycol], yy[scol], s_years, years, yy["ff"])
    return metrics.summarize(rv, dy, pnl, c, n_draws=500)


# ---------- report ----------

def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    """Write the evidence tables and the nested pruning results to runs/factor_review.md."""
    import logging

    from ust_rv.backtest import engine
    from ust_rv.config import factors, repo_path, settings, thresholds
    from ust_rv.panel import build_panel
    from ust_rv.scoring.scorecard import compute_scores

    logging.basicConfig(level=logging.ERROR)
    p = build_panel()
    s = compute_scores(p)
    lib = pd.DataFrame(factors()).set_index("id")
    zc, h = settings()["zscore"], settings()["horizon_days"]
    available = {k for k, m in p.meta.items() if m["status"] in ("ok", "stale")}
    lines = [
        "# Factor review: evidence tables",
        "",
        f"Generated by `python -m ust_rv.backtest.factor_review` on {pd.Timestamp.today():%Y-%m-%d}, data through {p.market_date:%Y-%m-%d}. "
        "IC = Spearman rank correlation of the factor's pressure with the realized 6m yield change (positive = behaves as signed). "
        "loo_delta_ic = composite IC (2009+) without the factor minus with it (positive = the composite is better without it). "
        "z_same_sign_5y = how one-sided the factor's z has been over 5 years (1 = always the same sign: a drift warning). "
        "Analysis and decisions: `FACTOR_REVIEW_PLAN.md`.",
        "",
    ]
    cols = ["factor", "sign", "weight", "ic_2005", "ic_2005-08", "ic_2009-15", "ic_2016-19", "ic_2020+", "ic_pit90",
            "eras_negative", "loo_delta_ic", "closest", "closest_corr", "z_same_sign_5y"]
    nested_rows = []
    for T in ("2y", "10y"):
        pit_dates = engine.windows(engine.pit_share(p, T))["PIT>=90%"]
        tab = factor_table(s.z, lib, s.theme_weights[T], s.dy_h[T], T, pit_dates, available, zc)
        loo_t = theme_leave_one_out(s.z, lib, s.theme_weights[T], s.dy_h[T], T, available, zc)
        lines += [f"## {T}: factors (sorted by IC since 2005)", "", _md(tab[cols].sort_values("ic_2005").reset_index()), "",
                  f"Theme leave-one-out (composite IC 2009+ change when the theme is removed): "
                  + ", ".join(f"{k} {v:+.3f}" for k, v in loo_t.items()), ""]
        for name, rule in PRUNING_RULES.items():
            comp, _ = nested_pruning(s.z, lib, s.theme_weights[T], s.dy_h[T], T, available, zc, h, rule)
            m = score_rule(comp, s.dy_h[T], p.yields, engine.TENOR_SPEC[T], *thresholds(T), start="2012-01-01")
            nested_rows.append({"rule": name, "tenor": T, "IC": m["ic"], "IC CI": f"[{m['ic_ci'][0]:.2f}, {m['ic_ci'][1]:.2f}]",
                                "hit": m["hit_rate"], "Sharpe": m["overlay_sharpe"],
                                "Sharpe CI": f"[{m['sharpe_ci'][0]:.2f}, {m['sharpe_ci'][1]:.2f}]", "bp/yr": m["pnl_bp_per_year"]})
    lines += ["## Selection-honest pruning test (2012+)", "",
              "Each January, every factor is judged only on targets realized before that January; the rule drops (or flips) factors and the "
              "composite is rebuilt for that year. `static` is the current model.", "", _md(pd.DataFrame(nested_rows)), ""]
    path = repo_path("runs") / "factor_review.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
