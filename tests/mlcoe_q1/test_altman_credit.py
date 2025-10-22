from __future__ import annotations

import pandas as pd
import pytest

from mlcoe_q1.credit import AltmanInputs, assign_rating_bucket, compute_altman_z, derive_altman_inputs


def test_compute_altman_z_matches_manual_calculation():
    inputs = AltmanInputs(
        total_assets=100.0,
        total_liabilities=60.0,
        current_assets=50.0,
        current_liabilities=20.0,
        retained_earnings=15.0,
        ebit=12.0,
        revenue=80.0,
        market_equity=90.0,
    )
    result = compute_altman_z(inputs)

    working_capital_ratio = (inputs.current_assets - inputs.current_liabilities) / inputs.total_assets
    retained_ratio = inputs.retained_earnings / inputs.total_assets
    ebit_ratio = inputs.ebit / inputs.total_assets
    market_ratio = inputs.market_equity / inputs.total_liabilities
    revenue_ratio = inputs.revenue / inputs.total_assets
    expected = 1.2 * working_capital_ratio + 1.4 * retained_ratio + 3.3 * ebit_ratio + 0.6 * market_ratio + revenue_ratio

    assert pytest.approx(expected) == result.z_score
    assert pytest.approx(working_capital_ratio) == result.working_capital_ratio
    assert pytest.approx(retained_ratio) == result.retained_earnings_ratio
    assert pytest.approx(ebit_ratio) == result.ebit_ratio
    assert pytest.approx(market_ratio) == result.market_equity_ratio
    assert pytest.approx(revenue_ratio) == result.revenue_ratio


@pytest.mark.parametrize(
    "score, expected",
    [
        (3.2, "investment_grade"),
        (2.6, "bbb"),
        (2.0, "bb"),
        (1.2, "b"),
        (0.5, "ccc"),
        (float("nan"), "unavailable"),
    ],
)
def test_assign_rating_bucket_thresholds(score, expected):
    assert assign_rating_bucket(score) == expected


def test_derives_inputs_with_market_and_book_equity():
    period = pd.Timestamp("2024-12-31")
    balance_sheet = pd.DataFrame(
        {
            period: {
                "Total Assets": 500.0,
                "Total Liabilities Net Minority Interest": 300.0,
                "Current Assets": 200.0,
                "Current Liabilities": 120.0,
                "Retained Earnings": 80.0,
                "Ordinary Shares Number": 10.0,
                "Stockholders Equity": 200.0,
            }
        }
    )
    income_statement = pd.DataFrame({period: {"EBIT": 60.0, "Total Revenue": 400.0}})

    inputs = derive_altman_inputs(
        balance_sheet,
        income_statement,
        period=period,
        market_equity=250.0,
    )
    assert inputs is not None
    assert inputs.market_equity == 250.0

    fallback_inputs = derive_altman_inputs(
        balance_sheet,
        income_statement,
        period=period,
        market_equity=None,
        fallback_equity=190.0,
    )
    assert fallback_inputs is not None
    assert fallback_inputs.market_equity == 190.0
