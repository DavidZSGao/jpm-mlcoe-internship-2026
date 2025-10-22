import numpy as np
import pandas as pd

from mlcoe_q1.credit.loan_pricing import (
    DEFAULT_SPREADS,
    LoanPricingParameters,
    compute_pricing,
)
from mlcoe_q1.pipelines import price_loans


def test_compute_pricing_components():
    params = LoanPricingParameters(
        risk_free_rate=0.04,
        leverage_target=0.5,
        leverage_factor=0.05,
        coverage_target=0.05,
        coverage_factor=0.4,
        scenario_factor=0.05,
        floor_rate=0.03,
    )

    result = compute_pricing(
        rating_bucket="bb",
        pred_total_assets=100.0,
        pred_equity=30.0,
        pred_net_income=2.0,
        scenario_quantile=0.1,
        params=params,
        spreads=DEFAULT_SPREADS,
        macro_values={"macro_policy_rate": 0.06},
        macro_config={
            "macro_policy_rate": {
                "baseline": "risk_free_rate",
                "multiplier": 0.5,
                "note": "Policy rate delta {delta:.3f} adds {adjustment:+.3f}",
            }
        },
    )

    assert np.isclose(result.base_rate, 0.04)
    assert result.base_spread == DEFAULT_SPREADS["bb"]
    # Leverage ratio = 0.7 -> (0.7 - 0.5) * 0.05 = 0.01
    assert np.isclose(result.leverage_adjustment, 0.01, atol=1e-6)
    # Net income margin 0.02 -> (0.05 - 0.02) * 0.4 = 0.012
    assert np.isclose(result.coverage_adjustment, 0.012, atol=1e-6)
    # Scenario adjustment (0.5 - 0.1) * 0.05 = 0.02
    assert np.isclose(result.scenario_adjustment, 0.02, atol=1e-6)
    # Macro adjustment: (0.06 - 0.04) * 0.5 = 0.01
    assert np.isclose(result.macro_adjustment, 0.01, atol=1e-6)
    expected_rate = 0.04 + DEFAULT_SPREADS["bb"] + 0.01 + 0.012 + 0.02 + 0.01
    assert np.isclose(result.recommended_rate, expected_rate)
    assert "Leverage" in result.notes
    assert "Margin" in result.notes
    assert "Downside" in result.notes
    assert "Policy rate" in result.notes
    assert set(result.macro_breakdown) == {"macro_policy_rate"}
    assert np.isclose(result.macro_breakdown["macro_policy_rate"], 0.01, atol=1e-6)


def test_price_loans_build_table_with_credit_metadata():
    scenarios = pd.DataFrame(
        {
            "ticker": ["GM", "GM"],
            "target_period": ["2022-12-31", "2022-12-31"],
            "scenario": ["baseline", "downside"],
            "pred_total_assets": [100.0, 100.0],
            "pred_equity": [45.0, 45.0],
            "pred_net_income": [5.0, 3.0],
            "scenario_quantile": [np.nan, 0.1],
            "macro_policy_rate": [0.04, 0.05],
        }
    )

    credit = pd.DataFrame(
        {
            "ticker": ["GM"],
            "period": ["2022-12-31"],
            "rating_bucket": ["bb"],
            "z_score": [2.0],
            "leverage": [0.55],
        }
    )

    params = LoanPricingParameters(risk_free_rate=0.04)
    macro_config = {"macro_policy_rate": {"baseline": 0.04, "multiplier": 0.4}}

    priced = price_loans.build_pricing_table(
        scenarios,
        credit,
        params=params,
        spreads=DEFAULT_SPREADS,
        macro_config=macro_config,
    )

    assert set(priced["scenario"]) == {"baseline", "downside"}
    downside = priced.loc[priced["scenario"] == "downside"].iloc[0]
    assert downside["rating_bucket"] == "bb"
    assert downside["leverage_ratio"] > 0.0
    assert downside["scenario_adjustment"] > 0.0
    assert downside["macro_adjustment"] > 0.0
    assert "macro_adj_macro_policy_rate" in downside
    assert downside["macro_adj_macro_policy_rate"] > 0
    assert downside["recommended_rate"] > downside["base_rate"]

    baseline = priced.loc[priced["scenario"] == "baseline"].iloc[0]
    assert baseline["scenario_adjustment"] == 0.0
    assert np.isclose(baseline["macro_adjustment"], 0.0, atol=1e-6)
    assert baseline["recommended_rate"] > baseline["base_rate"]

    summary = price_loans._summarise(priced)
    assert summary["downside"]["avg_macro_adjustment"] > 0
    assert summary["downside"]["avg_macro_adj_macro_policy_rate"] > 0


def test_price_scenarios_without_credit_uses_unavailable():
    scenarios = pd.DataFrame(
        {
            "ticker": ["MSFT"],
            "target_period": ["2021-12-31"],
            "scenario": ["baseline"],
            "pred_total_assets": [200.0],
            "pred_equity": [120.0],
            "pred_net_income": [20.0],
            "scenario_quantile": [0.8],
        }
    )

    priced = price_loans.build_pricing_table(
        scenarios,
        credit=None,
        params=LoanPricingParameters(risk_free_rate=0.03),
        spreads=DEFAULT_SPREADS,
    )

    row = priced.iloc[0]
    assert row["rating_bucket"] == "unavailable"
    assert row["scenario_adjustment"] < 0
    assert row["recommended_rate"] >= 0.03
    assert "macro_adjustment" not in row or np.isclose(row.get("macro_adjustment", 0.0), 0.0)
