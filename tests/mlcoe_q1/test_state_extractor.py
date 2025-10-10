"""Tests for state and income statement extraction helpers."""

from __future__ import annotations

import pandas as pd

from mlcoe_q1.utils import state_extractor as mod


def test_extract_income_metric_map(tmp_path) -> None:
    periods = [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    records = []
    for period in periods:
        records.extend(
            [
                {
                    "ticker": "TST",
                    "statement": "income_statement",
                    "line_item": "totalRevenue",
                    "period": period,
                    "value": 100.0,
                },
                {
                    "ticker": "TST",
                    "statement": "income_statement",
                    "line_item": "operatingIncome",
                    "period": period,
                    "value": 20.0,
                },
                {
                    "ticker": "TST",
                    "statement": "income_statement",
                    "line_item": "interestExpense",
                    "period": period,
                    "value": 5.0,
                },
                {
                    "ticker": "TST",
                    "statement": "income_statement",
                    "line_item": "netIncome",
                    "period": period,
                    "value": 12.0 + periods.index(period),
                },
            ]
        )
    df = pd.DataFrame.from_records(records)
    path = tmp_path / "sample.parquet"
    df.to_parquet(path, index=False)

    metrics = mod.extract_income_metric_map(path)
    assert set(metrics.keys()) == set(periods)
    latest = metrics[pd.Timestamp("2024-12-31")]
    assert latest.revenue == 100.0
    assert latest.ebit == 20.0
    assert latest.interest_expense == 5.0
    assert latest.net_income == 13.0
