"""Per-factor pressure, theme aggregation, composite and the master scorecard."""
from dataclasses import dataclass

import pandas as pd

from ust_rv.config import factors as load_factors
from ust_rv.config import settings, thresholds
from ust_rv.features.zscore import rolling_z
from ust_rv.panel import Panel
from ust_rv.scoring.composite import theme_pressure, weighted_average
from ust_rv.scoring.decision import hysteresis

TENORS = {"2y": "y2", "10y": "y10"}
Z_CHANGE_DAYS = 21


@dataclass
class Scores:
    z: pd.DataFrame  # grid x factor
    pressure: pd.DataFrame  # sign * z
    weights: dict[str, pd.Series]  # tenor -> factor weights
    theme_pressure: dict[str, pd.DataFrame]  # tenor -> grid x theme
    theme_z: dict[str, pd.DataFrame]
    theme_weights: dict[str, pd.Series]
    composite: pd.DataFrame  # grid x tenor
    rv: pd.DataFrame  # grid x tenor
    dy_h: pd.DataFrame  # forward 6m yield change, bp (NaN for the last `horizon` days)


def forward_change_bp(yields: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return (yields.shift(-horizon) - yields) * 100.0


def compute_scores(panel: Panel, factor_list: list[dict] | None = None) -> Scores:
    cfg = settings()
    zc = cfg["zscore"]
    lib = pd.DataFrame(factor_list if factor_list is not None else load_factors()).set_index("id")

    z = rolling_z(panel.values[lib.index], zc["window"], zc["min_periods"], zc["clip"])
    pressure = z.mul(lib["sign"], axis=1)

    tw = pd.DataFrame(cfg["theme_weights"]).T
    weights, tp, tz, tws, comp, rv = {}, {}, {}, {}, {}, {}
    for tenor in TENORS:
        w = lib[f"weight_{tenor}"].astype(float)
        weights[tenor] = w
        tp[tenor] = theme_pressure(pressure, w, lib["theme"])
        tz[tenor] = rolling_z(tp[tenor], zc["window"], zc["min_periods"], zc["clip"])
        tws[tenor] = tw[f"w{tenor}"].astype(float)[tz[tenor].columns]
        # composite re-standardized like the themes so decision thresholds are in sd units
        comp[tenor] = rolling_z(weighted_average(tz[tenor], tws[tenor]), zc["window"], zc["min_periods"], zc["clip"])
        rv[tenor] = hysteresis(comp[tenor], *thresholds(tenor))

    dy = forward_change_bp(panel.yields[list(TENORS.values())], cfg["horizon_days"]).rename(columns={v: k for k, v in TENORS.items()})
    return Scores(z, pressure, weights, tp, tz, tws, pd.DataFrame(comp), pd.DataFrame(rv), dy)


def sign_check(pressure: pd.Series, dy_h: pd.Series, horizon: int, window: int) -> float:
    """Spearman rho of pressure(t) vs realized dy_h(t) over the last `window` dates whose target is realized."""
    pairs = pd.concat([pressure, dy_h], axis=1).iloc[: len(pressure) - horizon].dropna().tail(window)
    if len(pairs) < window // 2:
        return float("nan")
    return float(pairs.iloc[:, 0].corr(pairs.iloc[:, 1], method="spearman"))


def scorecard(panel: Panel, scores: Scores, factor_list: list[dict] | None = None) -> pd.DataFrame:
    """Master scorecard for the latest grid date: one row per factor (Build_Plan 6.1)."""
    cfg = settings()
    lib = pd.DataFrame(factor_list if factor_list is not None else load_factors()).set_index("id")
    t = panel.date
    horizon, window, warn = cfg["horizon_days"], cfg["sign_check"]["window_days"], cfg["sign_check"]["warn_rho"]
    rows = []
    for fid, f in lib.iterrows():
        meta = panel.meta[fid]
        as_of = panel.as_of[fid].iloc[-1]
        age = (t - as_of).days if pd.notna(as_of) else None
        z = scores.z[fid]
        row = {
            "id": fid,
            "factor": f["name"],
            "theme": f["theme"],
            "value": panel.values[fid].iloc[-1],
            "as_of": as_of,
            "age_days": age,
            "fresh": age is not None and age <= meta["max_age_days"],
            "z": z.iloc[-1],
            "z_chg_1m": z.iloc[-1] - z.iloc[-1 - Z_CHANGE_DAYS],
            "pressure": scores.pressure[fid].iloc[-1],
            "status": meta["status"],
            "source": meta["label"] or "-",
            "fallback": meta["source"] == "fallback",
        }
        for tenor in TENORS:
            w = scores.weights[tenor][fid]
            row[f"w_{tenor}"] = w
            row[f"contrib_{tenor}"] = float("nan")
            if w > 0:
                rho = sign_check(scores.pressure[fid], scores.dy_h[tenor], horizon, window)
                row[f"rho_{tenor}"] = rho
                row[f"sign_{tenor}"] = "WARN" if rho < warn else ("PASS" if pd.notna(rho) else "-")
            else:
                row[f"rho_{tenor}"], row[f"sign_{tenor}"] = float("nan"), "-"
        rows.append(row)
    card = pd.DataFrame(rows).set_index("id")

    # Contribution = factor's share of its theme pressure: w * pressure / sum|w| over available theme factors.
    for tenor in TENORS:
        w = card[f"w_{tenor}"].where(card["pressure"].notna(), 0.0)
        den = w.groupby(card["theme"]).transform("sum")
        card[f"contrib_{tenor}"] = (card[f"w_{tenor}"] * card["pressure"] / den).where(card[f"w_{tenor}"] > 0)
    return card


def theme_table(scores: Scores) -> pd.DataFrame:
    """Latest theme pressure, theme z and weighted contribution to the composite, per tenor."""
    out = {}
    for tenor in TENORS:
        tz = scores.theme_z[tenor].iloc[-1]
        w = scores.theme_weights[tenor]
        avail = tz.notna()
        out[f"pressure_{tenor}"] = scores.theme_pressure[tenor].iloc[-1]
        out[f"z_{tenor}"] = tz
        out[f"W_{tenor}"] = w
        out[f"contrib_{tenor}"] = (w * tz / w[avail].abs().sum()).where(avail)
    return pd.DataFrame(out)
