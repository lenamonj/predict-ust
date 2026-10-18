import pandas as pd
import yaml

from ust_rv.backtest import calibrate, selection
from ust_rv.config import CONFIG_DIR


def test_write_settings_rewrites_only_decision_and_model_blocks(tmp_path):
    p = tmp_path / "settings.yaml"
    original = (CONFIG_DIR / "settings.yaml").read_text(encoding="utf-8")
    p.write_text(original, encoding="utf-8", newline="\n")
    calibrate.write_settings({"2y": ("phase1", 0.4, 0.1), "10y": ("regime", 0.75, 0.2)}, p)
    new = yaml.safe_load(p.read_text(encoding="utf-8"))
    old = yaml.safe_load(original)
    assert new["decision"] == {"2y": {"enter": 0.4, "exit": 0.1}, "10y": {"enter": 0.75, "exit": 0.2}}
    assert new["composite_model"] == {"2y": "phase1", "10y": "regime"}
    assert {k: v for k, v in new.items() if k not in ("decision", "composite_model")} == \
        {k: v for k, v in old.items() if k not in ("decision", "composite_model")}
    assert b"\r\n" not in p.read_bytes()


def test_best_rule_prefers_simplest_within_margin():
    idx = pd.MultiIndex.from_product([list(selection.RULES), ["2y", "10y"]], names=["rule", "tenor"])
    sharpe = {"static": 0.25, "a_shared": 0.27, "b_per_tenor_pair": 0.31, "c_per_tenor_model_pair": 0.10}
    summary = pd.DataFrame({"sharpe": [sharpe[r] for r, _ in idx]}, index=idx)
    r = selection.NestedResult(pd.DataFrame(), {}, {}, summary)
    assert selection.best_rule(r) == "a_shared"  # best is b (0.31); a (0.27) is simpler and within 0.05; static (0.25) is not
    summary.loc["b_per_tenor_pair", "sharpe"] = 0.29
    assert selection.best_rule(r) == "static"  # best is now b (0.29): static (0.25) is within 0.05
