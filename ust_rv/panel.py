"""Point-in-time daily panel: one column per factor (raw transformed value), plus as-of
release dates, pit flags, the 2y/10y yields and per-factor provenance metadata."""
import json
import logging
from dataclasses import dataclass

import pandas as pd

from ust_rv.config import factors as load_factors
from ust_rv.config import repo_path, settings
from ust_rv.models import fair_value
from ust_rv.providers.alfred import AlfredProvider
from ust_rv.providers.base import ParquetCache, ProviderError
from ust_rv.providers.bloomberg import BloombergProvider
from ust_rv.providers.fred import FredProvider
from ust_rv.providers.public import PublicProvider
from ust_rv.realtime import VintageError
from ust_rv.sources import Resolved, business_grid, resolve_source

log = logging.getLogger(__name__)

# Market series on the grid: curve points for targets (y2, y10) and overlay carry/roll (y1, y7, ff);
# equity and credit for the regime model (NASDAQCOM is FRED's only long daily equity index);
# 6m, 3y and 20y for the 6m-forward yields used by the forward-adjusted diagnostic.
YIELDS = {"y1": "DGS1", "y2": "DGS2", "y7": "DGS7", "y10": "DGS10", "ff": "DFF", "equity": "NASDAQCOM", "baa10y": "BAA10Y",
          "y6m": "DGS6MO", "y3": "DGS3", "y20": "DGS20"}


def default_providers(cache: ParquetCache | None = None) -> dict:
    cache = cache or ParquetCache()
    return {
        "fred": FredProvider(cache),
        "alfred": AlfredProvider(cache),
        "bloomberg": BloombergProvider(cache),
        "public": PublicProvider(cache),
    }


@dataclass
class Panel:
    values: pd.DataFrame  # grid x factor id
    as_of: pd.DataFrame
    pit: pd.DataFrame
    yields: pd.DataFrame  # y2, y10 in percent
    market_asof: pd.Timestamp
    meta: dict[str, dict]
    fair_value: fair_value.FairValueResult | None

    @property
    def date(self) -> pd.Timestamp:
        """Last grid date (values are forward-filled to it)."""
        return self.values.index[-1]

    @property
    def market_date(self) -> pd.Timestamp:
        """Last date with an actual 10y yield observation: the true 'data through' date."""
        return self.market_asof

    def save(self, directory=None) -> None:
        directory = directory or repo_path("cache") / "panel"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.date.strftime("%Y%m%d")
        wide = pd.concat(
            [self.values, self.as_of.add_suffix("__as_of"), self.pit.astype("boolean").add_suffix("__pit"), self.yields], axis=1
        )
        wide.to_parquet(directory / f"panel_{stamp}.parquet")
        with open(directory / "panel_meta.json", "w", encoding="utf-8") as f:
            json.dump({"panel_date": str(self.date.date()), "factors": self.meta}, f, indent=1, default=str)


def _resolve_factor(factor: dict, providers: dict, grid, fv: fair_value.FairValueResult | None) -> tuple[Resolved | None, str | None, list[str], str]:
    """Try primary then fallback. Returns (resolved, which, errors, status_if_failed)."""
    errors: list[str] = []
    only_bloomberg_failed = True
    for which in ("primary", "fallback"):
        src = factor["source"].get(which)
        if src is None:
            continue
        if src.get("provider") == "model":
            return Resolved(fv.frame, "model:fair_value"), which, errors, ""
        try:
            return resolve_source(src, factor["transform"], providers, grid), which, errors, ""
        except (ProviderError, VintageError) as exc:
            errors.append(str(exc))
            used = {i["provider"] for i in src.get("inputs", [src])}
            only_bloomberg_failed &= used == {"bloomberg"}
    return None, None, errors, "unavailable" if only_bloomberg_failed else "error"


def _effective_max_age(factor: dict, which: str | None) -> int:
    src = factor["source"].get(which) if which else None
    return (src or {}).get("max_age_days", factor["max_age_days"])


def build_panel(providers: dict | None = None, factor_list: list[dict] | None = None, end=None) -> Panel:
    cfg = settings()
    providers = providers or default_providers()
    factor_list = factor_list if factor_list is not None else load_factors()
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    grid = business_grid(cfg["calendar"]["start"], end)
    backtest_start = pd.Timestamp(cfg["backtest_start"])

    fv = None
    if any(f["source"]["primary"].get("provider") == "model" for f in factor_list):
        fv = fair_value.build(providers, grid, cfg["zscore"]["window"], cfg["zscore"]["min_periods"])

    values, as_of, pit, meta = {}, {}, {}, {}
    for f in factor_list:
        resolved, which, errors, failed_status = _resolve_factor(f, providers, grid, fv)
        for e in errors:
            log.warning("%s: %s", f["id"], e)
        info = {
            "name": f["name"],
            "theme": f["theme"],
            "source": which,
            "label": resolved.label if resolved else None,
            "stale": bool(resolved.stale) if resolved else False,
            "errors": errors + (resolved.errors if resolved else []),
            "max_age_days": _effective_max_age(f, which),
        }
        if resolved is None:
            values[f["id"]] = pd.Series(float("nan"), index=grid)
            as_of[f["id"]] = pd.Series(pd.NaT, index=grid, dtype="datetime64[ns]")
            pit[f["id"]] = pd.Series(pd.NA, index=grid, dtype="boolean")
            info.update(status=failed_status, last_as_of=None, first_valid=None, pit_share_bt=None)
        else:
            fr = resolved.frame
            values[f["id"]], as_of[f["id"]], pit[f["id"]] = fr["value"], fr["as_of"], fr["pit"]
            valid = fr["value"].dropna()
            bt = fr.loc[backtest_start:, "pit"].dropna()
            info.update(
                status="stale" if resolved.stale else "ok",
                last_as_of=fr["as_of"].dropna().iloc[-1] if not valid.empty else None,
                first_valid=valid.index[0] if not valid.empty else None,
                pit_share_bt=float(bt.astype(bool).mean()) if len(bt) else None,
            )
        meta[f["id"]] = info

    market = {k: resolve_source({"provider": "fred", "series": s}, {"type": "level"}, providers, grid).frame for k, s in YIELDS.items()}
    yields = pd.DataFrame({k: f["value"] for k, f in market.items()})
    market_asof = pd.Timestamp(market["y10"]["as_of"].dropna().iloc[-1])
    return Panel(pd.DataFrame(values), pd.DataFrame(as_of), pd.DataFrame(pit), yields, market_asof, meta, fv)
