import numpy as np
import pandas as pd


def rolling_z(x: pd.DataFrame | pd.Series, window: int, min_periods: int, clip: float):
    """Trailing rolling z-score, winsorized at +/-clip. NaN until `min_periods` valid observations."""
    roll = x.rolling(window, min_periods=min_periods)
    sd = roll.std().replace(0.0, np.nan)
    return ((x - roll.mean()) / sd).clip(-clip, clip)
