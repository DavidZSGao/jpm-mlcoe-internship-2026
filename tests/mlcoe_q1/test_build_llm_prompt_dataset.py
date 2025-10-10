"""Tests for the LLM prompt dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlcoe_q1.pipelines.build_llm_prompt_dataset import (
    build_prompt_dataset,
    main,
)


def _write_processed_fixture(path: Path, ticker: str) -> None:
    data = pd.DataFrame(
        [
            # Context period
            (ticker, "balance_sheet", "cash", "2023-12-31", 100.0),
            (ticker, "balance_sheet", "totalAssets", "2023-12-31", 1000.0),
            (ticker, "income_statement", "revenue", "2023-12-31", 250.0),
            # Target period
            (ticker, "balance_sheet", "cash", "2024-12-31", 120.0),
            (ticker, "balance_sheet", "totalAssets", "2024-12-31", 1100.0),
            (ticker, "income_statement", "revenue", "2024-12-31", 260.0),
        ],
        columns=["ticker", "statement", "line_item", "period", "value"],
    )
    data.to_parquet(path / f"{ticker}.parquet")


def test_build_prompt_dataset_emits_records(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    _write_processed_fixture(processed_root, "TEST")

    dataset = build_prompt_dataset(processed_root, ["balance_sheet", "income_statement"])

    assert len(dataset) == 1
    record = dataset.iloc[0]
    assert record["ticker"] == "TEST"
    assert record["context_period"] == "2023-12-31"
    assert "Balance Sheet" in record["prompt"]
    ground_truth = json.loads(record["ground_truth"])
    assert ground_truth["balance_sheet"]["totalAssets"] == 1100.0


def test_build_prompt_dataset_respects_max_prompts(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    _write_processed_fixture(processed_root, "AAA")
    _write_processed_fixture(processed_root, "BBB")

    dataset = build_prompt_dataset(
        processed_root,
        ["balance_sheet", "income_statement"],
        max_prompts=1,
    )

    assert len(dataset) == 1


def test_main_writes_json_output(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    processed_root.mkdir()
    _write_processed_fixture(processed_root, "CLI")

    output_path = tmp_path / "prompts.json"
    main(
        [
            "--processed-root",
            str(processed_root),
            "--output",
            str(output_path),
            "--log-level",
            "ERROR",
        ]
    )

    written = pd.read_json(output_path)
    assert not written.empty
