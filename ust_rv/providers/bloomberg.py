"""Bloomberg Desktop API via xbbg. Optional at runtime: with no terminal the cached history is
used (flagged stale); with no cache either, ProviderError so the panel falls through to the
factor's fallback source."""
import pandas as pd

from ust_rv.providers.base import Fetch, ParquetCache, cached_fetch, normalize_vintages

HISTORY_START = "1990-01-01"


class BloombergProvider:
    name = "bloomberg"

    def __init__(self, cache: ParquetCache | None = None):
        self.cache = cache or ParquetCache()

    def _fetch(self, ticker: str) -> pd.DataFrame:
        from xbbg import blp  # imported lazily: absent on non-terminal machines

        df = blp.bdh(ticker, "PX_LAST", HISTORY_START, pd.Timestamp.today().strftime("%Y-%m-%d"))
        if df.empty:
            raise ValueError(f"Bloomberg returned no data for {ticker}")
        s = df.iloc[:, 0].dropna()
        return normalize_vintages(pd.DataFrame({"ref_date": s.index, "release_date": s.index, "value": s.to_numpy()}))

    def get_vintages(self, ticker: str) -> Fetch:
        df, stale, err = cached_fetch(self.cache, self.name, ticker, lambda _: self._fetch(ticker))
        return Fetch(df, stale, err)
