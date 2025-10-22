"""Summarize forecaster evaluation artifacts with grouped error metrics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from mlcoe_q1.pipelines.summarize_forecaster_evaluation import summarize as _summarize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "reports/q1/artifacts/forecaster_evaluation.parquet"
        ),
        help="Path to the parquet artifact produced by evaluate_forecaster",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=["ticker", "mode"],
        help="Columns used to aggregate metrics (default: ticker+mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to persist the summary "
            "(extension determines format)"
        ),
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    return parser.parse_args(argv)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Evaluation artifact is missing columns: {missing}")


def summarize_metrics(df: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame:
    required = [
        "assets_mae",
        "equity_mae",
        "identity_gap",
    ]
    optional_income = "net_income_mae" in df.columns
    _validate_columns(df, list(group_by) + required)

    summary = _summarize(df, list(group_by))

    expected_columns = [
        "observations",
        "assets_mae_mean",
        "assets_mae_median",
        "assets_mae_max",
        "equity_mae_mean",
        "equity_mae_median",
        "equity_mae_max",
        "identity_gap_mean",
    ]
    if optional_income:
        expected_columns.extend(
            [
                "net_income_mae_mean",
                "net_income_mae_median",
                "net_income_mae_max",
            ]
        )

    missing_stats = [column for column in expected_columns if column not in summary.columns]
    if missing_stats:
        raise KeyError(
            "Summariser output is missing expected statistics: "
            + ", ".join(missing_stats)
        )

    return summary.sort_values(list(group_by)).reset_index(drop=True)


def _write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix in {".json"}:
        df.to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported output format for {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.evaluation.exists():
        raise FileNotFoundError(f"Evaluation artifact not found: {args.evaluation}")

    df = pd.read_parquet(args.evaluation)
    if df.empty:
        logging.warning("Evaluation artifact is empty; nothing to summarise")
        summary = df
    else:
        summary = summarize_metrics(df, args.group_by)

    if summary.empty:
        logging.info("No records available after summarisation")
    else:
        try:
            rendered = summary.to_markdown(index=False)
        except (
            ImportError,
            ModuleNotFoundError,
        ):  # pandas raises ImportError for tabulate
            rendered = summary.to_string(index=False)
        logging.info(
            "Summarised %d rows into %d group(s)\n%s",
            len(df),
            len(summary),
            rendered,
        )

    if args.output:
        _write_output(summary, args.output)
        logging.info("Summary written to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()

