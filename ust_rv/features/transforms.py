"""Factor transforms. Each takes a series at its native frequency (sorted, NaN-free)
and returns a trailing-only series on the same index. `periods` counts observations."""
import pandas as pd


def level(s: pd.Series) -> pd.Series:
    return s


def diff(s: pd.Series, periods: int) -> pd.Series:
    return s - s.shift(periods)


def pct_change(s: pd.Series, periods: int) -> pd.Series:
    return (s / s.shift(periods) - 1.0) * 100.0


def pct_change_annualized(s: pd.Series, periods: int, ppy: int) -> pd.Series:
    return ((s / s.shift(periods)) ** (ppy / periods) - 1.0) * 100.0


def mean_change(s: pd.Series, periods: int) -> pd.Series:
    """Average per-period change over the last `periods` observations."""
    return (s - s.shift(periods)) / periods


def accel(s: pd.Series, periods: int, ppy: int) -> pd.Series:
    """Short-horizon annualized rate minus the year-over-year rate."""
    return pct_change_annualized(s, periods, ppy) - pct_change(s, ppy)


def diff_vs_ma(s: pd.Series, window: int) -> pd.Series:
    return s - s.rolling(window).mean()


def momentum_12m_1m(s: pd.Series, long: int, short: int) -> pd.Series:
    """(s_t - s_{t-long}) - (s_t - s_{t-short}), i.e. the change from t-long to t-short."""
    return s.shift(short) - s.shift(long)


TRANSFORMS = {
    "level": level,
    "diff": diff,
    "pct_change": pct_change,
    "pct_change_annualized": pct_change_annualized,
    "mean_change": mean_change,
    "accel": accel,
    "diff_vs_ma": diff_vs_ma,
    "momentum_12m_1m": momentum_12m_1m,
}

_OPTIONS = {"type", "roll", "roll_how", "scale"}


def apply_transform(s: pd.Series, spec: dict) -> pd.Series:
    """Apply optional pre-smoothing (`roll` with `roll_how` mean|sum), the transform, then `scale`."""
    kind = spec["type"]
    if kind not in TRANSFORMS:
        raise ValueError(f"unknown transform {kind!r}")
    x = s.astype(float)
    if roll := spec.get("roll"):
        how = spec.get("roll_how", "mean")
        if how not in ("mean", "sum"):
            raise ValueError(f"unknown roll_how {how!r}")
        x = getattr(x.rolling(roll), how)()
    params = {k: v for k, v in spec.items() if k not in _OPTIONS}
    out = TRANSFORMS[kind](x, **params)
    return out * spec.get("scale", 1.0)
