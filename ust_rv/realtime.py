"""Point-in-time engine.

A vintage table has columns (ref_date, release_date, value): one row per (reference period,
date the value became public). Revised series carry several rows per ref_date.

`realtime_readings` turns a vintage table into the series of readings a trader saw: at each
release date, the transform is computed on the full vintage available that day and the
latest valid value is kept. Before a series' first vintage, the first vintage is used with
pseudo release dates (ref_date + median release lag) and flagged pit=False.
"""
import pandas as pd

from ust_rv.features.transforms import apply_transform

READING_COLUMNS = ["value", "ref_date", "pit"]


class VintageError(ValueError):
    """A vintage table the point-in-time engine cannot date (reported per factor, not raised to the caller)."""


def _empty_readings() -> pd.DataFrame:
    out = pd.DataFrame(columns=READING_COLUMNS, index=pd.DatetimeIndex([], name="release_date"))
    return out.astype({"value": float, "ref_date": "datetime64[ns]", "pit": bool})


def _is_unrevised(vt: pd.DataFrame) -> bool:
    """Exactly one row per ref_date and releases never go backwards as ref_date advances."""
    if not vt["ref_date"].is_unique:
        return False
    rel = vt.sort_values("ref_date")["release_date"]
    return bool((rel.cummax() == rel).all())


def _fast_readings(vt: pd.DataFrame, spec: dict, approx_lag_days: int | None) -> pd.DataFrame:
    vt = vt.sort_values("ref_date").reset_index(drop=True).assign(pit=True)
    # A first vintage holding many refs is a history backfill (ALFRED): give those refs pseudo release dates.
    first_vintage = vt["release_date"].min()
    batch = vt["release_date"] == first_vintage
    if batch.sum() > 1:
        lag = pd.Timedelta(days=approx_lag_days) if approx_lag_days is not None else median_release_lag(vt)
        pseudo = vt["ref_date"] + lag
        move = batch & (pseudo < first_vintage)
        vt.loc[move, "release_date"] = pseudo[move]
        vt.loc[move, "pit"] = False
    vt = vt.set_index("ref_date")
    tr = apply_transform(vt["value"], spec).dropna()
    out = pd.DataFrame(
        {"value": tr.to_numpy(), "ref_date": tr.index, "pit": vt["pit"].loc[tr.index].to_numpy()},
        index=pd.DatetimeIndex(vt["release_date"].loc[tr.index].to_numpy(), name="release_date"),
    )
    # several refs released together (batch): keep the latest ref
    return out[~out.index.duplicated(keep="last")]


def median_release_lag(vt: pd.DataFrame) -> pd.Timedelta:
    """Median of (first release - ref_date) over refs first published after the first vintage."""
    first_vintage = vt["release_date"].min()
    first_release = vt.groupby("ref_date")["release_date"].min()
    later = first_release[first_release > first_vintage]
    if later.empty:
        raise VintageError("cannot estimate release lag: every ref_date appears in the first vintage")
    return (later - later.index.to_series()).median()


def _vintage_readings(vt: pd.DataFrame, spec: dict, approx_lag_days: int | None) -> pd.DataFrame:
    wide = vt.pivot_table(index="ref_date", columns="release_date", values="value", aggfunc="last").sort_index()
    # A ref keeps its value in later vintages until revised.
    wide = wide.ffill(axis=1)

    rows = []
    for vintage in wide.columns:
        tr = apply_transform(wide[vintage].dropna(), spec).dropna()
        if not tr.empty:
            rows.append((vintage, tr.iloc[-1], tr.index[-1]))
    true_pit = pd.DataFrame(rows, columns=["release_date", "value", "ref_date"]).set_index("release_date")
    true_pit["pit"] = True

    first_vintage = wide.columns[0]
    lag = pd.Timedelta(days=approx_lag_days) if approx_lag_days is not None else median_release_lag(vt)
    tr0 = apply_transform(wide[first_vintage].dropna(), spec).dropna()
    pseudo_release = tr0.index + lag
    keep = pseudo_release < first_vintage
    approx = pd.DataFrame(
        {"value": tr0.to_numpy()[keep], "ref_date": tr0.index[keep], "pit": False},
        index=pd.DatetimeIndex(pseudo_release[keep], name="release_date"),
    )
    out = pd.concat([approx, true_pit]).sort_index()
    out.index.name = "release_date"
    return out


def realtime_readings(vt: pd.DataFrame, spec: dict, approx_lag_days: int | None = None) -> pd.DataFrame:
    """Readings indexed by release_date with columns value, ref_date, pit."""
    vt = vt.dropna(subset=["value"])
    if vt.empty:
        return _empty_readings()
    if _is_unrevised(vt):
        out = _fast_readings(vt, spec, approx_lag_days)
    else:
        out = _vintage_readings(vt, spec, approx_lag_days)
    return out[READING_COLUMNS].astype({"value": float, "pit": bool})


def to_grid(readings: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of join: on each grid date, the latest reading released on or before that date.
    Returns columns value, ref_date, as_of (release date), pit."""
    left = pd.DataFrame({"date": grid.astype("datetime64[ns]")})
    right = readings.reset_index().rename(columns={"release_date": "as_of"})
    right["as_of"] = right["as_of"].astype("datetime64[ns]")
    right = right.sort_values("as_of")
    merged = pd.merge_asof(left, right, left_on="date", right_on="as_of", direction="backward")
    return merged.set_index("date")[["value", "ref_date", "as_of", "pit"]]
