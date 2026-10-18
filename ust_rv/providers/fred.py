"""FRED latest values (unrevised daily market series). Release date = observation date."""
import os
from functools import cache

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from ust_rv.config import repo_path
from ust_rv.providers.base import Fetch, ParquetCache, cached_fetch, normalize_vintages

REVISION_OVERLAP_DAYS = 30


@cache
def fred_client() -> Fred:
    load_dotenv(repo_path("env_file"))
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set (expected in .env or the environment)")
    return Fred(api_key=key)


class FredProvider:
    name = "fred"

    def __init__(self, cache: ParquetCache | None = None, client: Fred | None = None):
        self.cache = cache or ParquetCache()
        self._client = client

    @property
    def client(self) -> Fred:
        return self._client or fred_client()

    def _fetch(self, series_id: str, cached: pd.DataFrame | None) -> pd.DataFrame:
        start = None
        if cached is not None and not cached.empty:
            start = cached["ref_date"].max() - pd.Timedelta(days=REVISION_OVERLAP_DAYS)
        s = self.client.get_series(series_id, observation_start=start)
        new = normalize_vintages(pd.DataFrame({"ref_date": s.index, "release_date": s.index, "value": s.to_numpy()}))
        if start is not None:
            new = pd.concat([cached[cached["ref_date"] < start], new], ignore_index=True)
        return new.dropna(subset=["value"]).reset_index(drop=True)

    def get_vintages(self, series_id: str) -> Fetch:
        df, stale, err = cached_fetch(self.cache, self.name, series_id, lambda c: self._fetch(series_id, c))
        return Fetch(df, stale, err)
