"""Baseline signals (Build_Plan 11.3), in RV space (+1 = long duration)."""
import numpy as np
import pandas as pd


def random_walk(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(0.0, index=index, name="random_walk")


def always_long(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(1.0, index=index, name="always_long")


def momentum(momentum_bp: pd.Series) -> pd.Series:
    """12m-1m yield momentum alone: yields trending up -> short duration."""
    return pd.Series(-np.sign(momentum_bp), index=momentum_bp.index, name="momentum")
