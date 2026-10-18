import pandas as pd


def weighted_average(values: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """sum(w*x) / sum(|w|) over the columns available on each date (renormalizes when data is missing)."""
    weights = weights[weights != 0]
    x = values[weights.index]
    num = x.mul(weights, axis=1).sum(axis=1, min_count=1)
    den = x.notna().mul(weights.abs(), axis=1).sum(axis=1)
    return num / den.where(den > 0)


def theme_pressure(pressure: pd.DataFrame, weights: pd.Series, themes: pd.Series) -> pd.DataFrame:
    """Per-theme weighted average of factor pressures. `weights` and `themes` are indexed by factor id."""
    out = {}
    for theme, ids in themes.groupby(themes).groups.items():
        w = weights[list(ids)]
        if (w != 0).any():
            out[theme] = weighted_average(pressure, w)
    return pd.DataFrame(out, index=pressure.index)
