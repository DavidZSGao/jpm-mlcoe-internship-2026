"""Tests for the LLM vs. forecaster comparison pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlcoe_q1.pipelines.compare_llm_and_forecaster import (
    compare_metrics,
    load_forecaster_evaluation,
    load_llm_metrics,
    summarize_comparison,
)


def _write_parquet(tmp_path: Path, name: str, data: list[dict[str, object]]) -> Path:
    path = tmp_path / name
    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)
    return path


def _write_json(tmp_path: Path, name: str, data: list[dict[str, object]]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_load_forecaster_evaluation_filters_mode(tmp_path: Path) -> None:
    artifact = _write_parquet(
        tmp_path,
        "forecaster.parquet",
        [
            {
                "ticker": "JPM",
                "target_period": "2021-12-31",
                "mode": "mlp",
                "assets_mae": 1.0,
                "equity_mae": 2.0,
                "identity_gap": 0.1,
                "net_income_mae": 3.0,
                "prev_period": "2021-09-30",
            },
            {
                "ticker": "JPM",
                "target_period": "2022-03-31",
                "mode": "bank_template",
                "assets_mae": 1.5,
                "equity_mae": 2.5,
                "identity_gap": 0.2,
                "net_income_mae": 3.5,
                "prev_period": "2021-12-31",
            },
        ],
    )

    filtered = load_forecaster_evaluation(artifact, mode="MLP")

    assert len(filtered) == 1
    assert filtered.iloc[0]["mode"] == "mlp"
    assert set(filtered.columns) == {
        "ticker",
        "target_period",
        "mode",
        "assets_mae",
        "equity_mae",
        "identity_gap",
        "net_income_mae",
        "prev_period",
    }


def test_load_llm_metrics_filters_model(tmp_path: Path) -> None:
    table = _write_json(
        tmp_path,
        "llm.json",
        [
            {
                "ticker": "JPM",
                "context_period": "2021-09-30",
                "target_period": "2021-12-31",
                "mae": 4.0,
                "mape": 0.1,
                "coverage": 0.8,
                "missing_items": 1,
                "extra_items": 0,
                "invalid_items": 0,
                "model": "baseline",
            },
            {
                "ticker": "JPM",
                "context_period": "2021-12-31",
                "target_period": "2022-03-31",
                "mae": 5.0,
                "mape": 0.2,
                "coverage": 0.6,
                "missing_items": 0,
                "extra_items": 1,
                "invalid_items": 0,
                "model": "other",
            },
        ],
    )

    filtered = load_llm_metrics(table, model_column="model", model="baseline")

    assert len(filtered) == 1
    assert filtered.iloc[0]["mae"] == 4.0
    assert "model" in filtered.columns


def test_compare_metrics_merges_on_ticker_period() -> None:
    forecaster_df = pd.DataFrame(
        [
            {
                "ticker": "JPM",
                "target_period": "2021-12-31",
                "mode": "mlp",
                "assets_mae": 1.0,
                "equity_mae": 2.0,
                "identity_gap": 0.1,
                "net_income_mae": 3.0,
            }
        ]
    )
    llm_df = pd.DataFrame(
        [
            {
                "ticker": "JPM",
                "context_period": "2021-09-30",
                "target_period": "2021-12-31",
                "mae": 4.0,
                "mape": 0.1,
                "coverage": 0.8,
                "missing_items": 1,
                "extra_items": 0,
                "invalid_items": 0,
            }
        ]
    )

    comparison = compare_metrics(forecaster_df, llm_df)

    assert list(comparison.columns) == [
        "ticker",
        "context_period",
        "target_period",
        "llm_mae",
        "llm_mape",
        "llm_coverage",
        "missing_items",
        "extra_items",
        "invalid_items",
        "forecaster_mode",
        "forecaster_assets_mae",
        "forecaster_equity_mae",
        "forecaster_identity_gap",
        "forecaster_net_income_mae",
    ]
    assert comparison.iloc[0]["forecaster_mode"] == "mlp"


def test_summarize_comparison_grouping() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "JPM",
                "llm_mae": 4.0,
                "llm_mape": 0.1,
                "llm_coverage": 0.8,
                "forecaster_assets_mae": 1.0,
                "forecaster_equity_mae": 2.0,
                "forecaster_identity_gap": 0.1,
                "forecaster_net_income_mae": 3.0,
            },
            {
                "ticker": "BAC",
                "llm_mae": 5.0,
                "llm_mape": 0.2,
                "llm_coverage": 0.6,
                "forecaster_assets_mae": 1.5,
                "forecaster_equity_mae": 2.5,
                "forecaster_identity_gap": 0.2,
                "forecaster_net_income_mae": 3.5,
            },
        ]
    )

    summary = summarize_comparison(df, group_by=["ticker"])

    assert len(summary) == 2
    assert set(summary.columns) == {
        "ticker",
        "records",
        "llm_mae_mean",
        "llm_mape_mean",
        "llm_coverage_mean",
        "forecaster_assets_mae_mean",
        "forecaster_equity_mae_mean",
        "forecaster_identity_gap_mean",
        "forecaster_net_income_mae_mean",
    }


def test_summarize_comparison_global_stats() -> None:
    df = pd.DataFrame(
        [
            {
                "llm_mae": 4.0,
                "llm_mape": 0.1,
                "llm_coverage": 0.8,
                "forecaster_assets_mae": 1.0,
                "forecaster_equity_mae": 2.0,
                "forecaster_identity_gap": 0.1,
            },
            {
                "llm_mae": 5.0,
                "llm_mape": 0.2,
                "llm_coverage": 0.6,
                "forecaster_assets_mae": 1.5,
                "forecaster_equity_mae": 2.5,
                "forecaster_identity_gap": 0.2,
            },
        ]
    )

    summary = summarize_comparison(df, group_by=[])

    assert summary.iloc[0]["records"] == 2
    assert pytest.approx(summary.iloc[0]["llm_mae_mean"]) == 4.5

