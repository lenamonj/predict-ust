"""How much to believe the backtest: placebo composites with random factor signs, the IC on
non-overlapping samples, and an exact decomposition of today's composite into factor terms."""
import numpy as np
import pandas as pd

from ust_rv.backtest import metrics
from ust_rv.features.zscore import rolling_z
from ust_rv.scoring.composite import theme_pressure, weighted_average


def composite_from_pressure(pressure: pd.DataFrame, weights: pd.Series, themes: pd.Series, theme_weights: pd.Series, zc: dict) -> pd.Series:
    """The Phase 1 composite pipeline: themes -> theme z -> weighted average -> composite z."""
    tz = rolling_z(theme_pressure(pressure, weights, themes), zc["window"], zc["min_periods"], zc["clip"])
    return rolling_z(weighted_average(tz, theme_weights[tz.columns]), zc["window"], zc["min_periods"], zc["clip"])


def placebo_ic(z: pd.DataFrame, weights: pd.Series, themes: pd.Series, theme_weights: pd.Series, dy: pd.Series, zc: dict,
               start: str, n: int = 200, seed: int = 0) -> np.ndarray:
    """IC of composites built with random factor signs (same factors, weights and pipeline).
    The spread shows how large an IC this structure can produce by chance."""
    rng = np.random.default_rng(seed)
    cols = weights.index[weights != 0]
    out = np.empty(n)
    for k in range(n):
        signs = pd.Series(rng.choice([-1.0, 1.0], size=len(cols)), index=cols)
        comp = composite_from_pressure(z[cols].mul(signs, axis=1), weights[cols], themes[cols], theme_weights, zc)
        out[k] = metrics.ic(comp.loc[start:], dy)
    return out


def nonoverlapping_ic(composite: pd.Series, dy: pd.Series, horizon: int, start: str) -> pd.DataFrame:
    """For each of the `horizon` possible start offsets, the IC on every horizon-th day (independent 6m windows)."""
    pairs = pd.concat([composite.rename("c"), dy.rename("dy")], axis=1, sort=True).loc[start:].dropna()
    rows = []
    for offset in range(horizon):
        s = pairs.iloc[offset::horizon]
        rows.append({"offset": offset, "n": len(s), "ic": metrics.ic(s["c"], s["dy"])})
    return pd.DataFrame(rows)


def theme_terms_at(theme_z: pd.DataFrame, theme_weights: pd.Series, composite_raw_hist: pd.Series, date: pd.Timestamp,
                   zc: dict) -> tuple[pd.Series, dict]:
    """Exact split of the composite z on `date` into one term per theme plus the composite-mean constant:
    raw = sum_T share_T * z_T, composite z = (raw - mu_c) / sd_c, so theme T adds share_T * z_T / sd_c."""
    tz = theme_z.loc[date]
    tz = tz[tz.notna()]
    share = theme_weights[tz.index] / theme_weights[tz.index].abs().sum()
    croll = composite_raw_hist.loc[:date].rolling(zc["window"], min_periods=zc["min_periods"])
    mu_c, sd_c = croll.mean().iloc[-1], croll.std().iloc[-1]
    z = float(np.clip((composite_raw_hist.loc[:date].iloc[-1] - mu_c) / sd_c, -zc["clip"], zc["clip"]))
    return share * tz / sd_c, {"composite_mean": float(-mu_c / sd_c), "composite_z": z}


def factor_terms_at(pressure: pd.DataFrame, weights: pd.Series, themes: pd.Series, theme_pressure_hist: pd.DataFrame,
                    theme_weights: pd.Series, composite_raw_hist: pd.Series, date: pd.Timestamp, zc: dict) -> tuple[pd.Series, dict]:
    """Exact split of the composite z on `date` into one term per factor plus constants (see composite_decomposition).
    Uses only data up to `date`."""
    tp = theme_pressure_hist.loc[:date]
    roll = tp.rolling(zc["window"], min_periods=zc["min_periods"])
    mu_t, sd_t = roll.mean().iloc[-1], roll.std().iloc[-1]
    croll = composite_raw_hist.loc[:date].rolling(zc["window"], min_periods=zc["min_periods"])
    mu_c, sd_c = croll.mean().iloc[-1], croll.std().iloc[-1]
    avail = tp.iloc[-1].notna() & sd_t.notna()
    share = theme_weights[avail[avail].index] / theme_weights[avail[avail].index].abs().sum()

    p = pressure.loc[date]
    w = weights[(weights > 0) & p.reindex(weights.index).notna() & themes.reindex(weights.index).isin(share.index)]
    den = w.groupby(themes[w.index]).transform("sum")
    contrib = w * p[w.index] / den  # factor's share of its theme pressure
    t = themes[w.index]
    terms = contrib * share[t].to_numpy() / sd_t[t].to_numpy() / sd_c
    z = float(np.clip((composite_raw_hist.loc[:date].iloc[-1] - mu_c) / sd_c, -zc["clip"], zc["clip"]))
    return terms, {"theme_means": float(-(share * mu_t[share.index] / sd_t[share.index]).sum() / sd_c),
                   "composite_mean": float(-mu_c / sd_c), "composite_z": z}


def composite_decomposition(card: pd.DataFrame, theme_pressure_hist: pd.DataFrame, theme_weights: pd.Series,
                            composite_raw_hist: pd.Series, tenor: str, zc: dict) -> tuple[pd.DataFrame, dict]:
    """Exact split of today's composite z into factor terms plus constants.

    theme z_T = (P_T - mu_T) / sd_T with P_T = sum of factor contributions in theme T;
    raw = sum_T (W_T / sum W) z_T; composite z = (raw - mu_c) / sd_c.
    So factor i adds (W_T / sum W) * contrib_i / sd_T / sd_c. The theme means and the composite mean
    are the remaining constant terms. Clipping at +/-3 (rare) shows up as a residual."""
    roll = theme_pressure_hist.rolling(zc["window"], min_periods=zc["min_periods"])
    mu_t, sd_t = roll.mean().iloc[-1], roll.std().iloc[-1]
    croll = composite_raw_hist.rolling(zc["window"], min_periods=zc["min_periods"])
    mu_c, sd_c = croll.mean().iloc[-1], croll.std().iloc[-1]
    today_p = theme_pressure_hist.iloc[-1]
    avail = today_p.notna() & sd_t.notna()
    w = theme_weights[avail[avail].index]
    share = w / w.abs().sum()

    c = card[(card[f"w_{tenor}"] > 0) & card[f"contrib_{tenor}"].notna() & card["theme"].isin(share.index)]
    terms = c[["factor", "theme", f"sign_{tenor}"]].rename(columns={f"sign_{tenor}": "sign_check"}).copy()
    terms["contribution"] = c[f"contrib_{tenor}"] * share[c["theme"]].to_numpy() / sd_t[c["theme"]].to_numpy() / sd_c
    theme_means = float(-(share * mu_t[share.index] / sd_t[share.index]).sum() / sd_c)
    composite_mean = float(-mu_c / sd_c)
    z_today = float(np.clip((composite_raw_hist.iloc[-1] - mu_c) / sd_c, -zc["clip"], zc["clip"]))
    residual = z_today - (terms["contribution"].sum() + theme_means + composite_mean)
    return terms.sort_values("contribution"), {"theme_means": theme_means, "composite_mean": composite_mean,
                                               "residual": float(residual), "composite_z": z_today}
