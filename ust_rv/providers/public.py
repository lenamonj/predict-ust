"""Free public feeds: TreasuryDirect auctions, CFTC Traders in Financial Futures, NY Fed
primary dealer positions. None are revised, so each returns one row per ref_date with the
release date implied by the publication schedule."""
import pandas as pd
import requests

from ust_rv.providers.base import Fetch, ParquetCache, cached_fetch, normalize_vintages

TIMEOUT = 90
TD_URL = "https://www.treasurydirect.gov/TA_WS/securities/search"
CFTC_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"  # TFF, futures only
NYFED_URL = "https://markets.newyorkfed.org/api/pd/get/{key}.json"

CFTC_CONTRACTS = {"TU": "042601", "TY": "043602", "TN": "043607"}
CFTC_RELEASE_LAG_DAYS = 3  # Tuesday positions published Friday
NYFED_RELEASE_LAG_DAYS = 8  # Wednesday positions published the following Thursday

TD_FIELDS = {
    "competitiveAuctionCloseDate": "auction_date",
    "securityType": "security_type",
    "originalSecurityTerm": "original_term",
    "offeringAmount": "offering_amount",
    "bidToCoverRatio": "bid_to_cover",
    "indirectBiddersCompetitiveParAwarded": "indirect_awarded",
    "subtotalCompetitiveParAwardedInsideOfOfferingAmount": "competitive_awarded",
    "inflationProtected": "tips",
    "floatingRate": "frn",
}


# ---------- parsers (pure, unit tested) ----------

def parse_treasurydirect(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records).reindex(columns=list(TD_FIELDS)).rename(columns=TD_FIELDS)
    df["auction_date"] = pd.to_datetime(df["auction_date"]).astype("datetime64[ns]")
    for c in ["offering_amount", "bid_to_cover", "indirect_awarded", "competitive_awarded"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["auction_date"]).sort_values("auction_date").reset_index(drop=True)


def auction_series(auctions: pd.DataFrame, term: str, metric: str) -> pd.DataFrame:
    """Per-auction series for one original term (originals and reopenings). Released on auction day."""
    a = auctions[(auctions["original_term"] == term) & (auctions["tips"] == "No") & (auctions["frn"] == "No")]
    if metric == "bid_to_cover":
        value = a["bid_to_cover"]
    elif metric == "indirect_share":
        value = a["indirect_awarded"] / a["competitive_awarded"] * 100.0
    else:
        raise ValueError(f"unknown auction metric {metric!r}")
    out = pd.DataFrame({"ref_date": a["auction_date"], "release_date": a["auction_date"], "value": value})
    return normalize_vintages(out.dropna(subset=["value"]).drop_duplicates("ref_date", keep="last"))


def coupon_issuance(auctions: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Gross nominal coupon issuance (Notes + Bonds, no TIPS/FRN) by auction month, $bn.
    The current month is incomplete and dropped; a month is released on its last day."""
    a = auctions[(auctions["tips"] == "No") & (auctions["frn"] == "No")]
    month = a["auction_date"].dt.to_period("M")
    total = a.groupby(month)["offering_amount"].sum() / 1e9
    total = total[total.index < today.to_period("M")]
    out = pd.DataFrame(
        {"ref_date": total.index.to_timestamp(), "release_date": total.index.to_timestamp(how="end").normalize(), "value": total.to_numpy()}
    )
    return normalize_vintages(out)


def parse_cftc(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    out = pd.DataFrame(
        {
            "report_date": pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).astype("datetime64[ns]"),
            "code": df["cftc_contract_market_code"],
            "lev_long": pd.to_numeric(df["lev_money_positions_long"]),
            "lev_short": pd.to_numeric(df["lev_money_positions_short"]),
            "open_interest": pd.to_numeric(df["open_interest_all"]),
        }
    )
    return out.sort_values("report_date").reset_index(drop=True)


def cftc_levnet_pct_oi(positions: pd.DataFrame, contracts: list[str]) -> pd.DataFrame:
    """Leveraged funds net (long - short) as % of open interest, summed across contracts."""
    codes = [CFTC_CONTRACTS[c] for c in contracts]
    p = positions[positions["code"].isin(codes)].groupby("report_date")[["lev_long", "lev_short", "open_interest"]].sum()
    value = (p["lev_long"] - p["lev_short"]) / p["open_interest"] * 100.0
    release = p.index + pd.Timedelta(days=CFTC_RELEASE_LAG_DAYS)
    return normalize_vintages(pd.DataFrame({"ref_date": p.index, "release_date": release, "value": value.to_numpy()}))


def parse_nyfed(payload: dict) -> pd.DataFrame:
    ts = pd.DataFrame(payload["pd"]["timeseries"])
    if ts.empty:
        raise ValueError("NY Fed returned an empty timeseries")
    ref = pd.to_datetime(ts["asofdate"])
    out = pd.DataFrame({"ref_date": ref, "release_date": ref + pd.Timedelta(days=NYFED_RELEASE_LAG_DAYS), "value": ts["value"]})
    return normalize_vintages(out).dropna(subset=["value"]).reset_index(drop=True)


# ---------- provider ----------

class PublicProvider:
    """Series ids:
    TD:BTC_10Y, TD:INDIRECT_10Y, TD:BTC_2Y, TD:COUPON_ISSUANCE,
    CFTC:LEVNET_TU, CFTC:LEVNET_TY_TN, NYFED:<keyid>"""

    name = "public"

    def __init__(self, cache: ParquetCache | None = None):
        self.cache = cache or ParquetCache()

    def _raw(self, key: str, fetch) -> tuple[pd.DataFrame, bool, str | None]:
        return cached_fetch(self.cache, self.name, key, lambda _: fetch())

    @staticmethod
    def _fetch_treasurydirect() -> pd.DataFrame:
        records = []
        for security_type in ("Note", "Bond"):
            r = requests.get(TD_URL, params={"securityType": security_type, "format": "json"}, timeout=TIMEOUT)
            r.raise_for_status()
            records += r.json()
        return parse_treasurydirect(records)

    @staticmethod
    def _fetch_cftc() -> pd.DataFrame:
        codes = ",".join(f"'{c}'" for c in CFTC_CONTRACTS.values())
        params = {
            "$select": "report_date_as_yyyy_mm_dd,cftc_contract_market_code,lev_money_positions_long,lev_money_positions_short,open_interest_all",
            "$where": f"cftc_contract_market_code in({codes})",
            "$limit": 50000,
        }
        r = requests.get(CFTC_URL, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return parse_cftc(r.json())

    @staticmethod
    def _fetch_nyfed(key: str) -> pd.DataFrame:
        r = requests.get(NYFED_URL.format(key=key), timeout=TIMEOUT)
        r.raise_for_status()
        return parse_nyfed(r.json())

    def get_vintages(self, series_id: str) -> Fetch:
        feed, _, name = series_id.partition(":")
        if feed == "TD":
            auctions, stale, err = self._raw("treasurydirect_auctions", self._fetch_treasurydirect)
            builders = {
                "BTC_10Y": lambda: auction_series(auctions, "10-Year", "bid_to_cover"),
                "INDIRECT_10Y": lambda: auction_series(auctions, "10-Year", "indirect_share"),
                "BTC_2Y": lambda: auction_series(auctions, "2-Year", "bid_to_cover"),
                "COUPON_ISSUANCE": lambda: coupon_issuance(auctions, pd.Timestamp.today()),
            }
            if name not in builders:
                raise ValueError(f"unknown TreasuryDirect series {series_id!r}")
            return Fetch(builders[name](), stale, err)
        if feed == "CFTC":
            contracts = {"LEVNET_TU": ["TU"], "LEVNET_TY_TN": ["TY", "TN"]}
            if name not in contracts:
                raise ValueError(f"unknown CFTC series {series_id!r}")
            positions, stale, err = self._raw("cftc_tff", self._fetch_cftc)
            return Fetch(cftc_levnet_pct_oi(positions, contracts[name]), stale, err)
        if feed == "NYFED":
            df, stale, err = self._raw(f"nyfed_{name}", lambda: self._fetch_nyfed(name))
            return Fetch(df, stale, err)
        raise ValueError(f"unknown public feed {series_id!r}")
