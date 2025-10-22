"""Tests for macro-conditioned scenario pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlcoe_q1.pipelines.simulate_macro_conditions import (
    apply_macro_scenarios,
    default_macro_scenarios,
    load_macro_scenarios,
)


def test_default_macro_scenarios_includes_baseline():
    scenarios = default_macro_scenarios()
    names = {scenario["name"] for scenario in scenarios}
    assert "baseline" in names
    assert any("downturn" in name for name in names)


def test_load_macro_scenarios_from_file(tmp_path: Path):
    config = {
        "scenarios": [
            {
                "name": "custom",
                "macro": {"gdp_growth": 0.02},
                "adjustments": {"pred_net_income": {"mode": "pct", "value": -0.1}},
                "description": "Custom scenario",
                "source": "unit_test",
            }
        ]
    }
    config_path = tmp_path / "scenarios.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    scenarios = load_macro_scenarios(config_path)
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "custom"
    assert scenarios[0]["macro"]["gdp_growth"] == 0.02
    assert scenarios[0]["source"] == "unit_test"


def test_apply_macro_scenarios_adjusts_metrics():
    df = pd.DataFrame(
        [
            {
                "ticker": "GM",
                "pred_total_assets": 100.0,
                "pred_equity": 20.0,
                "pred_net_income": 5.0,
            }
        ]
    )

    scenarios = [
        {
            "name": "baseline",
            "description": "No change",
            "macro": {"gdp_growth": 0.02},
            "adjustments": {},
            "source": "config",
        },
        {
            "name": "stress",
            "description": "Income down 40%, equity -10% absolute",
            "macro": {"gdp_growth": -0.01},
            "adjustments": {
                "pred_net_income": {"mode": "pct", "value": -0.4},
                "pred_equity": {"mode": "add", "value": -3.0},
            },
            "source": "config",
        },
    ]

    conditioned = apply_macro_scenarios(df, scenarios)
    assert len(conditioned) == 2

    baseline_row = conditioned[conditioned["scenario"] == "baseline"].iloc[0]
    assert pytest.approx(baseline_row["pred_net_income"]) == 5.0
    assert baseline_row["applied_adjustment_count"] == 0
    assert json.loads(baseline_row["macro_assumptions_json"]) == {"gdp_growth": 0.02}

    stress_row = conditioned[conditioned["scenario"] == "stress"].iloc[0]
    assert pytest.approx(stress_row["pred_net_income"]) == 3.0
    assert pytest.approx(stress_row["pred_equity"]) == 17.0
    adjustments = json.loads(stress_row["applied_adjustments_json"])
    assert set(adjustments) == {"pred_net_income", "pred_equity"}
    assert adjustments["pred_net_income"]["mode"] == "pct"
    assert adjustments["pred_net_income"]["base"] == 5.0
    assert pytest.approx(adjustments["pred_net_income"]["adjusted"]) == 3.0

