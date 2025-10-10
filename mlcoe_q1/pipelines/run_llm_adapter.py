"""Run a registered LLM adapter over the prompt dataset and capture responses."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlcoe_q1.llm.adapters import create_adapter


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="Prompt dataset path")
    parser.add_argument("--output", type=Path, required=True, help="Where to store responses")
    parser.add_argument(
        "--adapter",
        default="flan-t5",
        help="Adapter registry name (e.g. flan-t5)",
    )
    parser.add_argument("--model", help="Optional model identifier override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of prompts to process")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _build_prompt(row: pd.Series) -> str:
    statements = [str(x) for x in row["statements"]]
    schema = json.dumps({name: {} for name in statements}, indent=2)
    parts = [
        "You are a financial modelling assistant.",
        "Respond with JSON only using the structure shown below.",
        schema,
        str(row["prompt"]),
    ]
    return "\n\n".join(parts)


def _coerce_payload(text: str, statements: Sequence[str]) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = {}

    result: dict[str, dict[str, object]] = {}
    for name in statements:
        section = payload.get(name, {}) if isinstance(payload, dict) else {}
        if not isinstance(section, dict):
            section = {}
        cleaned: dict[str, object] = {}
        for key, value in section.items():
            cleaned[str(key)] = value
        result[str(name)] = cleaned
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    prompts = pd.read_parquet(args.prompts)
    if args.limit > 0:
        prompts = prompts.head(args.limit)

    adapter = create_adapter(
        args.adapter,
        model=args.model,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    records: list[dict[str, object]] = []
    for row in prompts.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        prompt_text = _build_prompt(row_series)
        statements = [str(x) for x in row_series["statements"]]
        completion = adapter.generate(
            prompt_text,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        payload = _coerce_payload(completion, statements)
        record = {k: getattr(row, k) for k in ["ticker", "context_period", "target_period"]}
        record.update(
            {
                "model": adapter.model_id,
                "adapter": adapter.name,
                "response": json.dumps(payload),
                "raw_response": completion,
            }
        )
        records.append(record)

    output_df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    logging.info("Wrote %d responses to %s", len(output_df), args.output)


if __name__ == "__main__":
    main()

