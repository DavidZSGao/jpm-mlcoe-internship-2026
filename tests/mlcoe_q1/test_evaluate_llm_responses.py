"""Tests for the LLM response evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlcoe_q1.pipelines.evaluate_llm_responses import (
    KEY_COLUMNS,
    evaluate_responses,
    main,
)


@pytest.fixture()
def prompt_dataset(tmp_path: Path) -> Path:
    data = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "context_period": "2022Q4",
                "target_period": "2023Q1",
                "statements": ["balance_sheet"],
                "ground_truth": json.dumps(
                    {"balance_sheet": {"cash": 120.0, "debt": 75.0}},
                    sort_keys=True,
                ),
                "prompt": "",
                "context": "",
            },
            {
                "ticker": "AAA",
                "context_period": "2023Q1",
                "target_period": "2023Q2",
                "statements": ["balance_sheet"],
                "ground_truth": json.dumps(
                    {"balance_sheet": {"cash": 140.0, "debt": 70.0}},
                    sort_keys=True,
                ),
                "prompt": "",
                "context": "",
            },
        ]
    )
    path = tmp_path / "prompts.parquet"
    data.to_parquet(path, index=False)
    return path


def test_evaluate_responses_scores_metrics(prompt_dataset: Path) -> None:
    prompts = pd.read_parquet(prompt_dataset)
    responses = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "context_period": "2022Q4",
                "target_period": "2023Q1",
                "model": "demo",
                "response": json.dumps(
                    {"balance_sheet": {"cash": 118, "debt": 80}},
                    sort_keys=True,
                ),
            }
        ]
    )

    metrics = evaluate_responses(prompts, responses, response_column="response", model_column="model")

    assert metrics.shape[0] == 1
    record = metrics.iloc[0]
    assert record["matched_items"] == 2
    assert pytest.approx(record["mae"], rel=1e-6) == 3.5
    assert record["coverage"] == pytest.approx(1.0)
    assert record["missing_items"] == 0
    assert record["extra_items"] == 0


def test_evaluate_responses_handles_missing_and_invalid(prompt_dataset: Path) -> None:
    prompts = pd.read_parquet(prompt_dataset)
    responses = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "context_period": "2023Q1",
                "target_period": "2023Q2",
                "response": json.dumps(
                    {
                        "balance_sheet": {
                            "cash": "140",
                            "debt": "not a number",
                            "equity": 50,
                        }
                    },
                    sort_keys=True,
                ),
            }
        ]
    )

    metrics = evaluate_responses(prompts, responses, response_column="response")

    record = metrics.iloc[0]
    assert record["matched_items"] == 1
    assert record["missing_items"] == 1  # debt could not be parsed
    assert record["extra_items"] == 1  # equity not in truth
    assert record["invalid_items"] == 1  # unparsable debt entry


def test_cli_round_trip(tmp_path: Path, prompt_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "context_period": "2022Q4",
                "target_period": "2023Q1",
                "response": json.dumps({"balance_sheet": {"cash": 120, "debt": 75}}),
            },
            {
                "ticker": "AAA",
                "context_period": "2023Q1",
                "target_period": "2023Q2",
                "response": json.dumps({"balance_sheet": {"cash": 135, "debt": 70}}),
            },
        ]
    )
    responses_path = tmp_path / "responses.csv"
    responses.to_csv(responses_path, index=False)

    output_path = tmp_path / "metrics.json"

    argv = [
        "--prompt-dataset",
        str(prompt_dataset),
        "--responses",
        str(responses_path),
        "--output",
        str(output_path),
        "--log-level",
        "DEBUG",
    ]

    main(argv)

    assert output_path.exists()
    saved = pd.read_json(output_path)
    assert list(saved.columns)[:3] == KEY_COLUMNS
    assert saved.shape[0] == 2

