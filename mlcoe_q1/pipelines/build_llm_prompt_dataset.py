"""Generate prompt/label pairs for LLM balance sheet forecasting experiments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


DEFAULT_STATEMENTS = ("balance_sheet", "income_statement")

PROMPT_TEMPLATE = """
You are assisting a lending team by forecasting corporate financial statements.
Company: {ticker}
Context period: {context_period}
Target period: {target_period}

Use the figures below to produce your forecast for the target period. For each statement,
return a JSON object whose keys are the line items and whose values are numeric forecasts
for the target period. Respond with JSON only.

{context_sections}
""".strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "mlcoe_q1/data/processed",
        help="Directory containing processed statement parquet files",
    )
    parser.add_argument(
        "--statements",
        nargs="*",
        default=list(DEFAULT_STATEMENTS),
        help="Financial statements to include in the prompt context",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=2,
        help="Minimum consecutive periods required to form a prompt",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        help="Optional cap on the number of prompts to emit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to persist the prompt dataset (csv/json/parquet)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    return parser.parse_args(argv)


def _load_processed_files(processed_root: Path) -> list[Path]:
    return sorted(
        p
        for p in processed_root.glob("*.parquet")
        if p.is_file() and p.name != "driver_features.parquet"
    )


def _format_statement_section(statement: str, items: dict[str, float], period: str) -> str:
    lines = [f"{statement.replace('_', ' ').title()} ({period})"]
    for name in sorted(items):
        value = items[name]
        formatted = f"{value:,.0f}" if pd.notna(value) else "NaN"
        lines.append(f"- {name}: {formatted}")
    return "\n".join(lines)


def _render_context(context: dict[str, dict[str, float]], period: str) -> str:
    sections = [
        _format_statement_section(statement, items, period)
        for statement, items in sorted(context.items())
    ]
    return "\n\n".join(sections)


def _statement_payload(
    df: pd.DataFrame, period: str, statement: str
) -> dict[str, float]:
    mask = (df["statement"] == statement) & (df["period"] == period)
    subset = df.loc[mask, ["line_item", "value"]].dropna(subset=["value"])
    return {row.line_item: float(row.value) for row in subset.itertuples(index=False)}


def _eligible_periods(
    df: pd.DataFrame, statements: Iterable[str]
) -> list[str]:
    ticker_periods = sorted(df["period"].unique())
    eligible: list[str] = []
    for period in ticker_periods:
        if all(_statement_payload(df, period, statement) for statement in statements):
            eligible.append(str(period))
    return eligible


def build_prompt_dataset(
    processed_root: Path,
    statements: Sequence[str],
    min_periods: int = 2,
    max_prompts: int | None = None,
) -> pd.DataFrame:
    """Create a dataframe of prompt/ground-truth pairs for LLM experiments."""

    files = _load_processed_files(processed_root)
    records: list[dict[str, object]] = []

    for file_path in files:
        df = pd.read_parquet(file_path)
        if df.empty:
            continue

        ticker = df["ticker"].iloc[0]
        ticker_df = df[df["ticker"] == ticker]
        periods = _eligible_periods(ticker_df, statements)
        if len(periods) < min_periods:
            continue

        for prev_period, target_period in zip(periods[:-1], periods[1:]):
            context_payload = {
                statement: _statement_payload(ticker_df, prev_period, statement)
                for statement in statements
            }
            target_payload = {
                statement: _statement_payload(ticker_df, target_period, statement)
                for statement in statements
            }

            if any(not payload for payload in context_payload.values()):
                continue
            if any(not payload for payload in target_payload.values()):
                continue

            context_sections = _render_context(context_payload, prev_period)
            prompt = PROMPT_TEMPLATE.format(
                ticker=ticker,
                context_period=prev_period,
                target_period=target_period,
                context_sections=context_sections,
            )

            record = {
                "ticker": ticker,
                "context_period": prev_period,
                "target_period": target_period,
                "statements": list(statements),
                "prompt": prompt,
                "context": json.dumps(context_payload, indent=2, sort_keys=True),
                "ground_truth": json.dumps(target_payload, indent=2, sort_keys=True),
            }
            records.append(record)

            if max_prompts is not None and len(records) >= max_prompts:
                return pd.DataFrame.from_records(records)

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

    if not args.processed_root.exists():
        raise FileNotFoundError(f"Processed directory not found: {args.processed_root}")
    if not args.statements:
        raise ValueError("At least one statement must be specified")
    if args.min_periods < 2:
        raise ValueError("min-periods must be >= 2")

    dataset = build_prompt_dataset(
        processed_root=args.processed_root,
        statements=args.statements,
        min_periods=args.min_periods,
        max_prompts=args.max_prompts,
    )

    if dataset.empty:
        logging.warning("No prompts generated. Check statement coverage and periods.")
    else:
        logging.info(
            "Generated %d prompt(s) for %d ticker(s)",
            len(dataset),
            dataset["ticker"].nunique(),
        )

    if args.output:
        _write_output(dataset, args.output)
        logging.info("Prompt dataset written to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()

