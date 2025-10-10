"""Emit heuristic LLM-style responses for the prompt dataset as a benchmarking baseline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


KEY_COLUMNS = ["ticker", "context_period", "target_period"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-dataset",
        type=Path,
        required=True,
        help="Table emitted by build_llm_prompt_dataset",
    )
    parser.add_argument(
        "--strategy",
        choices=["context_copy", "scaled_copy"],
        default="context_copy",
        help="Heuristic used to construct responses",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Multiplier applied when using the scaled_copy strategy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to persist the generated responses (csv/json/parquet)",
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


def _to_float_dict(payload: dict[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, raw in payload.items():
        if raw is None:
            continue
        try:
            values[key] = float(raw)
        except (TypeError, ValueError):
            continue
    return values


def _iter_records(prompts: pd.DataFrame) -> Iterable[dict[str, object]]:
    missing = [col for col in (*KEY_COLUMNS, "context", "statements") if col not in prompts.columns]
    if missing:
        raise ValueError(f"Prompt dataset missing required columns: {missing}")

    for row in prompts.itertuples(index=False):
        context_payload = json.loads(getattr(row, "context"))
        statements = list(getattr(row, "statements"))
        record = {k: getattr(row, k) for k in KEY_COLUMNS}
        record["context_payload"] = context_payload
        record["statements"] = statements
        yield record


def _context_copy_response(context_payload: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    return {
        statement: _to_float_dict(items)
        for statement, items in context_payload.items()
    }


def _scaled_copy_response(
    context_payload: dict[str, dict[str, object]],
    scale_factor: float,
) -> dict[str, dict[str, float]]:
    response: dict[str, dict[str, float]] = {}
    for statement, items in context_payload.items():
        response[statement] = {
            key: float(value) * scale_factor
            for key, value in _to_float_dict(items).items()
        }
    return response


def generate_responses(
    prompts: pd.DataFrame,
    strategy: str = "context_copy",
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """Construct heuristic responses aligned with the prompt dataset."""

    if strategy not in {"context_copy", "scaled_copy"}:
        raise ValueError(f"Unsupported strategy: {strategy}")

    records: list[dict[str, object]] = []
    for record in _iter_records(prompts):
        context_payload = record.pop("context_payload")
        if strategy == "context_copy":
            response_payload = _context_copy_response(context_payload)
        else:
            response_payload = _scaled_copy_response(context_payload, scale_factor)

        response_json = json.dumps(response_payload, sort_keys=True)
        records.append(
            {
                **{k: v for k, v in record.items() if k not in {"context_payload"}},
                "model": f"baseline:{strategy}",
                "response": response_json,
            }
        )

    return pd.DataFrame.from_records(records)


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
    responses = generate_responses(
        prompts,
        strategy=args.strategy,
        scale_factor=args.scale_factor,
    )

    _write_output(responses, args.output)
    logging.info(
        "Generated %d response(s) using %s strategy", len(responses), args.strategy
    )


if __name__ == "__main__":  # pragma: no cover
    main()

