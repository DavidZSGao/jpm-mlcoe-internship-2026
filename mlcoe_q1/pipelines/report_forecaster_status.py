"""Generate a Markdown status report from forecaster evaluation metrics."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlcoe_q1.pipelines.summarize_forecaster_eval import summarize_metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/forecaster_eval.parquet",
        help="Path to the parquet artifact produced by evaluate_forecaster",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=["ticker", "mode"],
        help="Columns used to aggregate metrics before generating the report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination for the Markdown report",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    return parser.parse_args(argv)


def _format_billions(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"${value / 1e9:.2f}B"


def _build_table(summary: pd.DataFrame) -> str:
    table_df = summary.copy()
    table_df["assets_mae_mean"] = table_df["assets_mae_mean"].apply(_format_billions)
    table_df["equity_mae_mean"] = table_df["equity_mae_mean"].apply(_format_billions)
    table_df["identity_gap_mean"] = table_df["identity_gap_mean"].apply(_format_billions)
    if "net_income_mae_mean" in table_df.columns:
        table_df["net_income_mae_mean"] = table_df["net_income_mae_mean"].apply(_format_billions)

    display_columns = [
        "ticker",
        "mode",
        "observations",
        "assets_mae_mean",
        "equity_mae_mean",
        "identity_gap_mean",
    ]
    if "net_income_mae_mean" in table_df.columns:
        display_columns.append("net_income_mae_mean")
    try:
        return table_df[display_columns].to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        return table_df[display_columns].to_string(index=False)


def _format_ticker_metric(row: pd.Series, column: str) -> str:
    return f"{row['ticker']} ({row['mode']}): {_format_billions(row[column])}"


def _build_highlights(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["- No evaluation records available to summarise."]

    highlights: list[str] = []

    top_equity = summary.sort_values("equity_mae_mean", ascending=False).head(3)
    equity_items = [
        _format_ticker_metric(row, "equity_mae_mean") for _, row in top_equity.iterrows()
    ]
    highlights.append(
        "- Highest equity MAE tickers: " + ", ".join(equity_items)
    )

    top_assets = summary.sort_values("assets_mae_mean", ascending=False).head(3)
    asset_items = [
        _format_ticker_metric(row, "assets_mae_mean") for _, row in top_assets.iterrows()
    ]
    highlights.append(
        "- Highest assets MAE tickers: " + ", ".join(asset_items)
    )

    if "net_income_mae_mean" in summary.columns:
        income_rows = summary.dropna(subset=["net_income_mae_mean"])
        if not income_rows.empty:
            top_income = income_rows.sort_values("net_income_mae_mean", ascending=False).head(3)
            income_items = [
                _format_ticker_metric(row, "net_income_mae_mean") for _, row in top_income.iterrows()
            ]
            highlights.append(
                "- Highest net income MAE tickers: " + ", ".join(income_items)
            )

    bank_rows = summary[summary["mode"].str.contains("bank", case=False, na=False)]
    if not bank_rows.empty:
        worst_bank = bank_rows.sort_values("equity_mae_mean", ascending=False).iloc[0]
        highlights.append(
            "- Bank coverage relies on templates for %d tickers; worst equity MAE is %s."
            % (
                len(bank_rows),
                _format_ticker_metric(worst_bank, "equity_mae_mean"),
            )
        )

    high_identity = summary[summary["identity_gap_mean"].abs() > 1e9]
    if high_identity.empty:
        highlights.append(
            "- Accounting identity gaps remain below $1B across evaluated tickers."
        )
    else:
        identity_items = [
            _format_ticker_metric(row, "identity_gap_mean")
            for _, row in high_identity.iterrows()
        ]
        highlights.append(
            "- Identity gap outliers (> $1B): " + ", ".join(identity_items)
        )

    return highlights


def render_report(summary: pd.DataFrame) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = ["# Forecaster Status Report", "", f"_Generated: {generated}_", ""]

    if summary.empty:
        lines.append("No evaluation records are available.")
        return "\n".join(lines)

    lines.extend(["## Aggregate Metrics", "", _build_table(summary), ""])
    lines.extend(["## Highlights", ""])
    lines.extend(_build_highlights(summary))

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.evaluation.exists():
        raise FileNotFoundError(f"Evaluation artifact not found: {args.evaluation}")

    df = pd.read_parquet(args.evaluation)
    if df.empty:
        logging.warning("Evaluation artifact is empty; no status report generated")
        summary = pd.DataFrame()
    else:
        summary = summarize_metrics(df, args.group_by)

    report = render_report(summary)
    logging.info("Generated status report for %d ticker group(s)", len(summary))
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        logging.info("Report written to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()

