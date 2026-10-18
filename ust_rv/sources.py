"""Resolve a factor source spec (single or multi-input) into a daily grid frame
with columns value, as_of, pit."""
from dataclasses import dataclass, field

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from ust_rv.providers.base import DataProvider
from ust_rv.realtime import realtime_readings, to_grid


def business_grid(start, end) -> pd.DatetimeIndex:
    """Business days excluding US federal holidays (the Treasury market calendar, less Good Friday nuance)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end).normalize()
    holidays = USFederalHolidayCalendar().holidays(start, end)
    return pd.bdate_range(start, end, freq="C", holidays=holidays).astype("datetime64[ns]")


@dataclass
class Resolved:
    frame: pd.DataFrame  # value, as_of, pit on the grid
    label: str
    stale: bool = False
    errors: list[str] = field(default_factory=list)


def _inputs(source: dict) -> list[dict]:
    return source["inputs"] if "inputs" in source else [source]


def source_label(source: dict) -> str:
    sep = {"diff": " - ", "ratio": " / "}.get(source.get("combine"), "")
    return sep.join(f"{i['provider']}:{i['series']}" for i in _inputs(source))


def input_on_grid(inp: dict, default_transform: dict, providers: dict[str, DataProvider], grid) -> tuple[pd.DataFrame, bool, str | None]:
    fetch = providers[inp["provider"]].get_vintages(inp["series"])
    vt = fetch.vintages
    if lag := inp.get("release_lag_days"):
        vt = vt.assign(release_date=vt["release_date"] + pd.Timedelta(days=lag))
    readings = realtime_readings(vt, inp.get("transform", default_transform), inp.get("approx_lag_days"))
    if pit_from := inp.get("pit_from"):
        # history published retroactively (e.g. a report launched later with backfilled data)
        readings.loc[readings.index < pd.Timestamp(pit_from), "pit"] = False
    return to_grid(readings, grid), fetch.stale, fetch.error


def resolve_source(source: dict, default_transform: dict, providers: dict[str, DataProvider], grid) -> Resolved:
    """Raises ProviderError if any input cannot be fetched."""
    parts, stale, errors = [], False, []
    for inp in _inputs(source):
        g, s, err = input_on_grid(inp, default_transform, providers, grid)
        parts.append(g)
        stale |= s
        if err:
            errors.append(err)
    if len(parts) == 1:
        frame = parts[0][["value", "as_of", "pit"]]
    else:
        a, b = parts
        combine = source["combine"]
        if combine == "diff":
            value = a["value"] - b["value"]
        elif combine == "ratio":
            value = a["value"] / b["value"]
        else:
            raise ValueError(f"unknown combine {combine!r}")
        frame = pd.DataFrame(
            {
                "value": value,
                "as_of": pd.concat([a["as_of"], b["as_of"]], axis=1).min(axis=1, skipna=False),
                "pit": a["pit"].astype("boolean") & b["pit"].astype("boolean"),
            }
        )
    frame = frame.astype({"value": float, "pit": "boolean"})
    frame.loc[frame["value"].isna(), ["as_of", "pit"]] = pd.NA
    return Resolved(frame, source_label(source), stale, errors)
