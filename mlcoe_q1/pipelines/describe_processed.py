"""Summarize processed financial statements using pandas utilities."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from mlcoe_q1.utils.statement_loader import load_all_processed


SUMMARY_COLUMNS = [
    "ticker",
    "statement",
    "line_item",
    "periods",
    "first_period",
    "last_period",
    "latest_value",
    "mean_value",
]


def _summarize_single(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create a tidy summary for a single ticker's processed dataframe."""

    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    ordered = df.sort_values("period")
    grouped = (
        ordered.groupby(["statement", "line_item"], dropna=False)
        .agg(
            periods=("period", "nunique"),
            first_period=("period", "min"),
            last_period=("period", "max"),
            latest_value=("value", "last"),
            mean_value=("value", "mean"),
        )
        .reset_index()
    )

    grouped.insert(0, "ticker", ticker)
    return grouped[SUMMARY_COLUMNS]


def summarize_processed(root: Path, tickers: Iterable[str]) -> pd.DataFrame:
    """Load processed parquet statements and compute descriptive statistics."""

    frames = []
    for ticker, df in load_all_processed(root, tickers).items():
        summary = _summarize_single(df, ticker)
        if not summary.empty:
            frames.append(summary)

    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def _infer_format(path: Path, preferred: str | None) -> str:
    if preferred:
        return preferred

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix == ".csv":
        return "csv"
    return "json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed",
        help="Directory containing processed parquet statements.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional subset of tickers. Defaults to all parquet files in --processed-root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports/q1/artifacts/processed_summary.json",
        help="Location to write the summary table.",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "json"],
        help="Explicit output format. Defaults to inference from --output suffix.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (e.g. INFO, DEBUG).",
    )
    parser.add_argument(
        "--statement",
        choices=["balance_sheet", "income_statement", "cashflow_statement"],
        help="Optional statement filter for the output table.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    processed_root = args.processed_root
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = [p.stem.upper() for p in processed_root.glob("*.parquet")]

    logging.info("Building pandas summary for %d tickers", len(tickers))
    summary = summarize_processed(processed_root, tickers)

    if args.statement:
        summary = summary[summary["statement"] == args.statement]

    if summary.empty:
        logging.warning("No processed statements found for requested tickers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_format = _infer_format(args.output, args.format)

    if output_format == "parquet":
        summary.to_parquet(args.output, index=False)
    elif output_format == "csv":
        summary.to_csv(args.output, index=False)
    else:
        payload = summary.to_dict(orient="records")
        with open(args.output, "w") as fh:
            json.dump(payload, fh, default=str, indent=2)

    logging.info("Summary written to %s as %s", args.output, output_format)


if __name__ == "__main__":  # pragma: no cover
    main()

