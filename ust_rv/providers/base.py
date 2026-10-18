"""Provider interface and parquet cache.

Every provider returns a vintage table (ref_date, release_date, value). Unrevised data has
one row per ref_date; ALFRED data has one row per (ref_date, vintage)."""
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

from ust_rv.config import repo_path

log = logging.getLogger(__name__)

VINTAGE_COLUMNS = ["ref_date", "release_date", "value"]


class ProviderError(RuntimeError):
    pass


@dataclass
class Fetch:
    vintages: pd.DataFrame
    stale: bool = False
    error: str | None = None


class DataProvider(Protocol):
    name: str

    def get_vintages(self, series_id: str) -> Fetch: ...


def normalize_vintages(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ref_date": pd.to_datetime(df["ref_date"]).astype("datetime64[ns]"),
            "release_date": pd.to_datetime(df["release_date"]).astype("datetime64[ns]"),
            "value": pd.to_numeric(df["value"], errors="coerce").astype(float),
        }
    )
    return out.sort_values(["ref_date", "release_date"]).reset_index(drop=True)


class ParquetCache:
    def __init__(self, root: Path | None = None):
        self.root = root or repo_path("cache")

    def path(self, provider: str, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        return self.root / provider / f"{safe}.parquet"

    def read(self, provider: str, key: str) -> pd.DataFrame | None:
        p = self.path(provider, key)
        return pd.read_parquet(p) if p.exists() else None

    def write(self, provider: str, key: str, df: pd.DataFrame) -> None:
        p = self.path(provider, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)

    def fetched_today(self, provider: str, key: str) -> bool:
        p = self.path(provider, key)
        return p.exists() and date.fromtimestamp(p.stat().st_mtime) == date.today()


def cached_fetch(
    cache: ParquetCache,
    provider: str,
    key: str,
    fetch: Callable[[pd.DataFrame | None], pd.DataFrame],
    refresh: bool = False,
) -> tuple[pd.DataFrame, bool, str | None]:
    """Return (frame, stale, error). Refetches at most once per day unless `refresh`.
    `fetch` receives the cached frame (or None) so it can update incrementally.
    On fetch failure the cached frame is returned flagged stale; with no cache, ProviderError."""
    cached = cache.read(provider, key)
    if cached is not None and not refresh and cache.fetched_today(provider, key):
        return cached, False, None
    try:
        fresh = fetch(cached)
    except Exception as exc:  # network, HTTP and API errors all degrade the same way
        msg = f"{provider}:{key} fetch failed: {exc}"
        if cached is None:
            # The panel decides severity: a factor with no working source is marked error or unavailable.
            log.warning(msg)
            raise ProviderError(msg) from exc
        log.warning(msg + " (using cache)")
        return cached, True, msg
    cache.write(provider, key, fresh)
    return fresh, False, None
