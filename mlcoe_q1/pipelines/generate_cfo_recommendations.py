"""Generate CFO recommendations by combining forecaster and LLM evaluations."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--forecaster-eval",
        type=Path,
        required=True,
        help="Parquet produced by evaluate_forecaster",
    )
    parser.add_argument(
        "--llm-eval",
        type=Path,
        help="Optional LLM evaluation table produced by evaluate_llm_responses",
    )
    parser.add_argument("--output", type=Path, required=True, help="Markdown destination")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--top", type=int, default=3, help="Number of tickers to highlight")
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "forecaster_eval": Path,
            "llm_eval": Path,
            "output": Path,
        },
    )


def _format_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            value = row[col]
            if pd.isna(value):
                cells.append("N/A")
            elif isinstance(value, float):
                cells.append(f"{value:,.2e}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _summarise_forecaster(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("Forecaster evaluation table is empty")
    summary = (
        df.groupby("ticker")
        .agg(
            observations=("ticker", "size"),
            assets_mae_mean=("assets_mae", "mean"),
            equity_mae_mean=("equity_mae", "mean"),
            net_income_mae_mean=("net_income_mae", "mean"),
        )
        .reset_index()
        .sort_values("equity_mae_mean", ascending=False)
    )
    return summary


def _summarise_llm(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    grouped = (
        df.groupby("ticker")
        .agg(
            mae_mean=("mae", "mean"),
            coverage_mean=("coverage", "mean"),
            invalid_mean=("invalid_items", "mean"),
        )
        .reset_index()
        .sort_values("mae_mean", ascending=False)
    )
    return grouped


def _ticker_recommendation(row: pd.Series) -> str:
    guidance: list[str] = []
    if row.get("equity_mae_mean", 0.0) and row["equity_mae_mean"] > 1e11:
        guidance.append("Prioritise capital structure reviews; bank ensemble calibration should be revisited.")
    elif row.get("equity_mae_mean", 0.0) and row["equity_mae_mean"] > 5e10:
        guidance.append("Monitor equity drift and validate leverage drivers before extending credit limits.")
    else:
        guidance.append("Equity forecasts are stable; maintain current underwriting thresholds.")

    net_income_mae = row.get("net_income_mae_mean")
    if pd.notna(net_income_mae) and net_income_mae > 1e10:
        guidance.append("Earnings volatility is material; stress-test cash flow coverage scenarios.")
    elif pd.notna(net_income_mae):
        guidance.append("Earnings forecasts align with history; proceed with standard profitability checks.")
    else:
        guidance.append("Net income not instrumented; supplement with analyst consensus before decisions.")

    return " ".join(guidance)


def _llm_recommendation(row: pd.Series) -> str:
    coverage = row.get("coverage_mean", 0.0)
    if coverage < 0.5:
        return "Low coverage — rely on deterministic model for critical ratios."
    if row.get("mae_mean", 0.0) > 1e11:
        return "High error — treat LLM output as qualitative context only."
    return "Coverage acceptable; integrate LLM narratives with quantitative checks."


def build_report(
    forecaster_summary: pd.DataFrame,
    llm_summary: pd.DataFrame | None,
    top_n: int,
) -> str:
    sections = ["# CFO Recommendations", ""]

    overview_cols = ["ticker", "observations", "assets_mae_mean", "equity_mae_mean", "net_income_mae_mean"]
    overview = forecaster_summary[overview_cols].copy()
    sections.append("## Forecast Accuracy Snapshot")
    sections.append(_format_markdown_table(overview))
    sections.append("")

    sections.append("## Ticker-Level Guidance")
    highlighted = forecaster_summary.head(top_n)
    for row in highlighted.itertuples(index=False):
        guidance = _ticker_recommendation(pd.Series(row._asdict()))
        sections.append(f"- **{row.ticker}** — {guidance}")
    sections.append("")

    if llm_summary is not None:
        sections.append("## LLM Diagnostic")
        sections.append(_format_markdown_table(llm_summary))
        sections.append("")
        for row in llm_summary.head(top_n).itertuples(index=False):
            sections.append(f"- **{row.ticker}** — {_llm_recommendation(pd.Series(row._asdict()))}")
        sections.append("")

    sections.append("## Next Steps")
    sections.append(
        "Focus on lowering residual bank MAE through driver enrichment and leverage audits while "
        "using the LLM runs for qualitative sanity checks rather than hard targets."
    )

    return "\n".join(sections)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    forecaster_summary = _summarise_forecaster(args.forecaster_eval)
    llm_summary = _summarise_llm(args.llm_eval) if args.llm_eval else None

    report = build_report(forecaster_summary, llm_summary, args.top)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    logging.info("Recommendations saved to %s", args.output)


if __name__ == "__main__":
    main()

