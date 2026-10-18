"""Decision rule (Build_Plan Section 10). Composite pressure is in yield space (positive =
yields higher); the RV score is in price space (+1 = buy = yields lower)."""
import numpy as np
import pandas as pd


def hysteresis(pressure: pd.Series, enter: float, exit: float) -> pd.Series:
    """+1 buy / -1 sell / 0 neutral, holding the prior state inside the band exit <= |p| < enter.
    NaN until the first valid pressure; a NaN pressure later holds the prior state."""
    out = np.full(len(pressure), np.nan)
    state = np.nan
    for i, p in enumerate(pressure.to_numpy()):
        if np.isnan(p):
            out[i] = state
            continue
        if np.isnan(state):
            state = 0.0
        if p >= enter:
            state = -1.0
        elif p <= -enter:
            state = 1.0
        elif abs(p) < exit:
            state = 0.0
        out[i] = state
    return pd.Series(out, index=pressure.index, name="rv")


def curve_read(rv_2y: int, rv_10y: int) -> str:
    """Display-only 2s10s read: long 2y relative to 10y is a steepener."""
    if rv_2y > rv_10y:
        return "steepener"
    if rv_2y < rv_10y:
        return "flattener"
    return "neutral"


def next_threshold(pressure: float, rv: int, enter: float, exit: float) -> tuple[str, float]:
    """The nearest threshold that would change the signal, and the pressure move needed to reach it."""
    if rv == 0:
        return ("sell (+%.2f)" % enter, enter - pressure) if pressure >= 0 else ("buy (-%.2f)" % enter, -enter - pressure)
    if rv == -1:  # short: exits when pressure falls inside the exit band
        return "exit to 0 (+%.2f)" % exit, exit - pressure
    return "exit to 0 (-%.2f)" % exit, -exit - pressure
