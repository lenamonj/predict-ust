from ust_rv.config import repo_path, settings


def test_settings_load():
    s = settings()
    assert s["horizon_days"] == 126
    assert s["zscore"]["window"] == 2520
    assert set(s["theme_weights"]) == {"growth", "labor", "inflation", "fed", "supply", "valuation", "momentum", "global"}


def test_repo_path_resolves_to_repo_root():
    assert repo_path("cache").parent.joinpath("pyproject.toml").exists()
