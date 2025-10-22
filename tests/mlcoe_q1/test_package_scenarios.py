import numpy as np
import pandas as pd

from mlcoe_q1.pipelines import package_scenarios


def test_build_scenarios_quantile_and_point_usage():
    df = pd.DataFrame(
        {
            "ticker": ["GM"],
            "prev_period": ["2020-12-31"],
            "target_period": ["2021-12-31"],
            "horizon": [1],
            "mode": ["mlp"],
            "distribution": ["gaussian"],
            "mc_strategy": ["gaussian_head"],
            "mc_sample_count": [128],
            "true_total_assets": [100.0],
            "true_equity": [40.0],
            "true_net_income": [5.0],
            "pred_total_assets": [102.0],
            "pred_equity": [38.0],
            "pred_net_income": [4.5],
            "identity_gap": [0.01],
            "pred_total_assets_q10": [95.0],
            "pred_total_assets_q90": [110.0],
            "pred_equity_q10": [32.0],
            "pred_equity_q90": [45.0],
            "pred_net_income_q10": [2.0],
            "pred_net_income_q90": [7.0],
            "identity_gap_q10": [-0.05],
            "identity_gap_q90": [0.07],
        }
    )

    scenarios = {
        "baseline": None,
        "downside": 0.1,
        "upside": 0.9,
    }

    packaged = package_scenarios.build_scenarios(df, scenarios)
    assert set(packaged["scenario"]) == {"baseline", "downside", "upside"}

    baseline = packaged.set_index("scenario").loc["baseline"]
    assert baseline["pred_total_assets"] == 102.0
    assert np.isnan(baseline["scenario_quantile"])
    assert baseline["scenario_source_assets"] == "point_estimate"

    downside = packaged.set_index("scenario").loc["downside"]
    assert downside["pred_equity"] == 32.0
    assert downside["scenario_source_equity"] == "quantile"
    assert downside["scenario_quantile"] == 0.1

    upside = packaged.set_index("scenario").loc["upside"]
    assert upside["pred_net_income"] == 7.0
    assert upside["scenario_source_net_income"] == "quantile"


def test_build_scenarios_falls_back_when_quantile_missing():
    df = pd.DataFrame(
        {
            "ticker": ["MSFT"],
            "prev_period": ["2020-12-31"],
            "target_period": ["2021-12-31"],
            "horizon": [1],
            "mode": ["mlp"],
            "pred_total_assets": [250.0],
            "pred_equity": [120.0],
            "identity_gap": [0.0],
            "true_total_assets": [245.0],
            "true_equity": [118.0],
        }
    )

    packaged = package_scenarios.build_scenarios(df, {"baseline": 0.5})
    assert len(packaged) == 1
    row = packaged.iloc[0]
    assert row["pred_total_assets"] == 250.0
    assert np.isnan(row["scenario_quantile"]) or row["scenario_quantile"] == 0.5
    assert row["scenario_source_assets"] == "point_estimate"


def test_parse_scenarios_parses_defaults():
    parsed = package_scenarios._parse_scenarios("baseline:,downside:0.1,upside:0.9")
    assert parsed == {"baseline": None, "downside": 0.1, "upside": 0.9}

    parsed_empty = package_scenarios._parse_scenarios("  ")
    assert parsed_empty == {"baseline": None}
