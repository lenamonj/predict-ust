from functools import cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).parent
REPO_ROOT = CONFIG_DIR.parents[1]


def _load(name: str):
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def settings() -> dict:
    return _load("settings.yaml")


@cache
def factors() -> list[dict]:
    return _load("factors.yaml")


@cache
def bloomberg_tickers() -> dict:
    return _load("tickers_bloomberg.yaml")


def thresholds(tenor: str) -> tuple[float, float]:
    d = settings()["decision"][tenor]
    return float(d["enter"]), float(d["exit"])


def composite_model(tenor: str) -> str:
    return settings()["composite_model"][tenor]


def repo_path(key: str) -> Path:
    return REPO_ROOT / settings()["paths"][key]
