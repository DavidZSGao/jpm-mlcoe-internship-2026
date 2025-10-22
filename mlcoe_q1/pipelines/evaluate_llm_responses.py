"""Evaluate LLM forecast responses against the prompt dataset ground truth."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


KEY_COLUMNS = ["ticker", "context_period", "target_period"]


@dataclass(frozen=True)
class RecordEvaluation:
    """Container for per-record evaluation statistics."""

    total_items: int
    matched_items: int
    mae: float | None
    mape: float | None
    missing_items: int
    extra_items: int
    invalid_items: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-dataset",
        type=Path,
        required=True,
        help="Path to the prompt dataset emitted by build_llm_prompt_dataset",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="Table of model responses containing forecast payloads",
    )
    parser.add_argument(
        "--response-column",
        default="response",
        help="Column in the responses table containing the model output JSON",
    )
    parser.add_argument(
        "--model-column",
        default="model",
        help="Optional column name indicating the model identifier",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to persist per-record metrics (csv/json/parquet)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    return parser.parse_args(argv)


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format for {path}")


def _parse_json(value: object) -> dict[str, object]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise TypeError("Expected JSON string or dict payload")


def _to_float(value: object) -> float:
    if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().replace(",", "")
        if not normalized:
            raise ValueError("Empty string")
        return float(normalized)
    raise ValueError(f"Unsupported value type {type(value)!r}")


def _statement_items(payload: dict[str, object]) -> tuple[dict[str, float], int]:
    items: dict[str, float] = {}
    invalid = 0
    for name, raw_value in payload.items():
        try:
            items[name] = _to_float(raw_value)
        except ValueError:
            invalid += 1
    return items, invalid


def _evaluate_record(
    ground_truth: dict[str, dict[str, float]],
    response_payload: dict[str, dict[str, object]],
) -> RecordEvaluation:
    total_items = 0
    matched_items = 0
    missing_items = 0
    extra_items = 0
    invalid_items = 0
    absolute_errors: list[float] = []
    percentage_errors: list[float] = []

    for statement, truth_items in ground_truth.items():
        total_items += len(truth_items)
        predicted_raw = response_payload.get(statement, {})
        if not isinstance(predicted_raw, dict):
            invalid_items += 1
            continue

        predicted_items, invalid_count = _statement_items(predicted_raw)
        invalid_items += invalid_count

        for name, truth_value in truth_items.items():
            if name not in predicted_items:
                missing_items += 1
                continue

            matched_items += 1
            prediction = predicted_items[name]
            absolute_errors.append(abs(prediction - truth_value))
            if truth_value != 0:
                percentage_errors.append(abs((prediction - truth_value) / truth_value))

        extra_items += len({k for k in predicted_items if k not in truth_items})

    mae = float(np.mean(absolute_errors)) if absolute_errors else float("nan")
    mape = float(np.mean(percentage_errors)) if percentage_errors else float("nan")

    return RecordEvaluation(
        total_items=total_items,
        matched_items=matched_items,
        mae=mae,
        mape=mape,
        missing_items=missing_items,
        extra_items=extra_items,
        invalid_items=invalid_items,
    )


def evaluate_responses(
    prompts: pd.DataFrame,
    responses: pd.DataFrame,
    response_column: str,
    model_column: str | None = None,
) -> pd.DataFrame:
    """Compute error metrics for model responses."""

    missing_cols = [c for c in KEY_COLUMNS if c not in prompts.columns]
    if missing_cols:
        raise ValueError(f"Prompt dataset missing required columns: {missing_cols}")

    if response_column not in responses.columns:
        raise ValueError(f"Responses missing column: {response_column}")

    if model_column is not None and model_column not in responses.columns:
        model_column = None

    merged = prompts.merge(responses, on=KEY_COLUMNS, how="inner", suffixes=("", "_response"))
    if merged.empty:
        raise ValueError("No overlapping prompt/response records found")

    evaluations: list[dict[str, object]] = []

    has_seed = "seed" in responses.columns
    has_adapter = "adapter" in responses.columns

    for row in merged.itertuples(index=False):
        ground_truth = _parse_json(getattr(row, "ground_truth"))
        response_payload = _parse_json(getattr(row, response_column))

        # ensure nested dict structure
        if not isinstance(ground_truth, dict) or not isinstance(response_payload, dict):
            raise TypeError("Ground truth and responses must be JSON objects")

        evaluation = _evaluate_record(ground_truth, response_payload)
        record = {k: getattr(row, k) for k in KEY_COLUMNS}
        record.update(
            {
                "statements": getattr(row, "statements"),
                "mae": evaluation.mae,
                "mape": evaluation.mape,
                "total_items": evaluation.total_items,
                "matched_items": evaluation.matched_items,
                "coverage": (
                    evaluation.matched_items / evaluation.total_items
                    if evaluation.total_items
                    else np.nan
                ),
                "missing_items": evaluation.missing_items,
                "extra_items": evaluation.extra_items,
                "invalid_items": evaluation.invalid_items,
            }
        )
        if model_column is not None:
            record[model_column] = getattr(row, model_column)
        if has_seed:
            record["seed"] = getattr(row, "seed", None)
        if has_adapter:
            record["adapter"] = getattr(row, "adapter", None)
        evaluations.append(record)

    return pd.DataFrame.from_records(evaluations)


def _summarize(metrics: pd.DataFrame, model_column: str | None) -> pd.DataFrame:
    group_keys: Iterable[str]
    if model_column and model_column in metrics.columns:
        group_keys = [model_column]
    else:
        group_keys = []

    grouped = metrics.groupby(list(group_keys), dropna=False) if group_keys else [((), metrics)]

    summary_rows: list[dict[str, object]] = []

    if isinstance(grouped, list):
        iter_groups = grouped
    else:
        iter_groups = grouped

    for key, frame in iter_groups:
        if group_keys:
            key_values = key if isinstance(key, tuple) else (key,)
            summary_row = dict(zip(group_keys, key_values))
        else:
            summary_row = {}

        summary_row.update(
            {
                "records": len(frame),
                "mean_mae": frame["mae"].dropna().mean(),
                "mean_mape": frame["mape"].dropna().mean(),
                "mean_coverage": frame["coverage"].mean(),
                "total_items": frame["total_items"].sum(),
                "matched_items": frame["matched_items"].sum(),
                "missing_items": frame["missing_items"].sum(),
                "extra_items": frame["extra_items"].sum(),
                "invalid_items": frame["invalid_items"].sum(),
            }
        )
        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows)


def _write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported output format for {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    prompts = _load_table(args.prompt_dataset)
    responses = _load_table(args.responses)

    metrics = evaluate_responses(
        prompts,
        responses,
        response_column=args.response_column,
        model_column=args.model_column,
    )

    summary = _summarize(metrics, args.model_column)

    logging.info("Evaluated %d response(s)", len(metrics))
    logging.info("Summary:\n%s", summary.to_string(index=False))

    if args.output:
        _write_output(metrics, args.output)
        logging.info("Per-record metrics written to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()

