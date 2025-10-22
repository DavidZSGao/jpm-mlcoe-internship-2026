from __future__ import annotations

import pandas as pd

from mlcoe_q1.pipelines.build_credit_rating_dataset import build_credit_dataset


PERIOD = pd.Timestamp("2024-12-31")


BALANCE_SHEETS = {
    "AAA": pd.DataFrame(
        {
            PERIOD: {
                "Total Assets": 500.0,
                "Total Liabilities Net Minority Interest": 250.0,
                "Current Assets": 220.0,
                "Current Liabilities": 120.0,
                "Retained Earnings": 90.0,
                "Ordinary Shares Number": 12.0,
                "Stockholders Equity": 260.0,
            }
        }
    ),
    "BBB": pd.DataFrame(
        {
            PERIOD: {
                "Total Assets": 400.0,
                "Total Liabilities Net Minority Interest": 260.0,
                "Current Assets": 140.0,
                "Current Liabilities": 130.0,
                "Retained Earnings": 40.0,
                "Ordinary Shares Number": 8.0,
                "Stockholders Equity": 140.0,
            }
        }
    ),
}

INCOME_STATEMENTS = {
    "AAA": pd.DataFrame({PERIOD: {"EBIT": 70.0, "Total Revenue": 420.0}}),
    "BBB": pd.DataFrame({PERIOD: {"EBIT": 20.0, "Total Revenue": 250.0}}),
}


def statement_loader(ticker: str):
    return BALANCE_SHEETS[ticker], INCOME_STATEMENTS[ticker]


def price_fetcher(ticker: str, period: pd.Timestamp):
    if ticker == "AAA":
        return 25.0
    return None


def test_build_credit_dataset_uses_price_when_available():
    dataset = build_credit_dataset(
        ["AAA", "BBB"],
        min_year=2024,
        max_year=2024,
        statement_loader=statement_loader,
        price_fetcher=price_fetcher,
    )

    assert set(dataset["ticker"]) == {"AAA", "BBB"}
    assert (dataset.loc[dataset["ticker"] == "AAA", "equity_source"].iloc[0]) == "market"
    assert (dataset.loc[dataset["ticker"] == "BBB", "equity_source"].iloc[0]) == "book"
    assert dataset.loc[dataset["ticker"] == "AAA", "market_equity_ratio"].iloc[0] > 0
    assert dataset.loc[dataset["ticker"] == "BBB", "market_equity_ratio"].iloc[0] > 0
    # Ensure Altman Z-score is computed and rating bucket present
    assert "z_score" in dataset.columns
    assert "rating_bucket" in dataset.columns
