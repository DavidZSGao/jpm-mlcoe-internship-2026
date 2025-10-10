"""Tests for the LLM baseline response generator CLI and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mlcoe_q1.pipelines import generate_llm_baseline_responses as baseline


def _sample_prompt_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "GM",
                "context_period": "2022",
                "target_period": "2023",
                "statements": ["balance_sheet"],
                "context": json.dumps(
                    {"balance_sheet": {"total_assets": 100.0, "total_equity": 40.0}}
                ),
            }
        ]
    )


def test_generate_responses_context_copy() -> None:
    prompts = _sample_prompt_dataframe()
    responses = baseline.generate_responses(prompts, strategy="context_copy")

    for column in ["ticker", "context_period", "target_period", "model", "response", "statements"]:
        assert column in responses.columns
    assert responses["model"].iloc[0] == "baseline:context_copy"
    payload = json.loads(responses["response"].iloc[0])
    assert payload["balance_sheet"]["total_assets"] == 100.0
    assert payload["balance_sheet"]["total_equity"] == 40.0


def test_generate_responses_scaled_copy() -> None:
    prompts = _sample_prompt_dataframe()
    responses = baseline.generate_responses(prompts, strategy="scaled_copy", scale_factor=1.1)

    payload = json.loads(responses["response"].iloc[0])
    assert payload["balance_sheet"]["total_assets"] == pytest.approx(110.0)
    assert payload["balance_sheet"]["total_equity"] == pytest.approx(44.0)


def test_cli_writes_expected_file(tmp_path: Path) -> None:
    prompts = _sample_prompt_dataframe()
    prompt_path = tmp_path / "prompts.parquet"
    output_path = tmp_path / "responses.json"
    prompts.to_parquet(prompt_path, index=False)

    baseline.main(
        [
            "--prompt-dataset",
            str(prompt_path),
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    responses = pd.read_json(output_path)
    assert len(responses) == 1
