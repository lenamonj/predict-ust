"""ALFRED point-in-time vintages. Release date = realtime_start.

Uses the FRED observations endpoint directly with pagination: the API caps a response at
100,000 rows and fredapi does not paginate, which silently truncates heavily revised series
(NFCI re-estimates its full history weekly: ~930k rows)."""
import os

import pandas as pd
import requests
from dotenv import load_dotenv

from ust_rv.config import repo_path
from ust_rv.providers.base import Fetch, ParquetCache, cached_fetch, normalize_vintages

URL = "https://api.stlouisfed.org/fred/series/observations"
PAGE = 100_000
TIMEOUT = 300


def _api_key() -> str:
    load_dotenv(repo_path("env_file"))
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not set (expected in .env or the environment)")
    return key


def fetch_all_releases(series_id: str, get=requests.get) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": _api_key(),
        "file_type": "json",
        "realtime_start": "1776-07-04",
        "realtime_end": "9999-12-31",
        "limit": PAGE,
        "offset": 0,
    }
    rows, expected = [], None
    while expected is None or len(rows) < expected:
        r = get(URL, params=params, timeout=TIMEOUT)
        payload = r.json()
        if "error_message" in payload:
            raise ValueError(f"FRED error for {series_id}: {payload['error_message']}")
        r.raise_for_status()
        expected = payload["count"]
        page = payload["observations"]
        if not page:
            break
        rows += page
        params["offset"] += len(page)
    if len(rows) != expected:
        raise ValueError(f"ALFRED {series_id}: got {len(rows)} rows, API reported {expected}")
    df = pd.DataFrame(rows, columns=["realtime_start", "date", "value"])
    return normalize_vintages(df.rename(columns={"date": "ref_date", "realtime_start": "release_date"}))


class AlfredProvider:
    name = "alfred"

    def __init__(self, cache: ParquetCache | None = None, get=requests.get):
        self.cache = cache or ParquetCache()
        self._get = get

    def _fetch(self, series_id: str) -> pd.DataFrame:
        df = fetch_all_releases(series_id, self._get)
        if df.empty:
            raise ValueError(f"ALFRED returned no vintages for {series_id}")
        return df

    def get_vintages(self, series_id: str) -> Fetch:
        df, stale, err = cached_fetch(self.cache, self.name, series_id, lambda _: self._fetch(series_id))
        return Fetch(df, stale, err)
