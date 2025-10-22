"""Compile a consolidated Strategic Lending executive summary."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--forecaster-summary",
        type=Path,
        default=REPO_ROOT
        / "reports/q1/artifacts/forecaster_evaluation_summary.parquet",
        help="Evaluation summary parquet produced by summarize_forecaster_evaluation",
    )
    parser.add_argument(
        "--scenario-summary",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/scenario_reasonableness.parquet",
        help="Scenario diagnostics parquet produced by assess_scenario_reasonableness",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/forecaster_calibration.parquet",
        help="Calibration diagnostics parquet from analyze_forecaster_calibration",
    )
    parser.add_argument(
        "--macro-scenarios",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/macro_conditioned_scenarios.parquet",
        help="Macro-conditioned scenario parquet produced by simulate_macro_conditions",
    )
    parser.add_argument(
        "--llm-seed-summary",
        type=Path,
        default=REPO_ROOT
        / "reports/q1/artifacts/llm_benchmarks/summary_by_model.parquet",
        help="Seed-aggregated LLM benchmark metrics (summary_by_model.parquet)",
    )
    parser.add_argument(
        "--loan-pricing-summary",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/loan_pricing_summary.json",
        help="JSON summary emitted by price_loans",
    )
    parser.add_argument(
        "--credit-metadata",
        type=Path,
        default=REPO_ROOT / "data/credit_ratings/altman_features.json",
        help="Metadata JSON produced by build_credit_rating_dataset",
    )
    parser.add_argument(
        "--risk-summary",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/risk_warnings_summary.json",
        help="JSON summary emitted by extract_risk_warnings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports/q1/status/executive_summary.md",
        help="Destination path for the consolidated Markdown summary",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Strategic Lending Executive Summary",
        help="Heading used at the top of the summary",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level for status output")
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "forecaster_summary": Path,
            "scenario_summary": Path,
            "calibration_report": Path,
            "macro_scenarios": Path,
            "llm_seed_summary": Path,
            "loan_pricing_summary": Path,
            "credit_metadata": Path,
            "risk_summary": Path,
            "output": Path,
        },
    )


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if path and path.exists():
        return pd.read_parquet(path)
    logging.info("Skipping missing parquet: %s", path)
    return None


def _load_json(path: Path) -> dict | None:
    if path and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    logging.info("Skipping missing JSON: %s", path)
    return None


def _format_float(value: float | None, *, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def _format_percent(value: float | None, *, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def _weighted_average(series: pd.Series, weights: pd.Series | None) -> float | None:
    mask = series.notna()
    if weights is not None:
        mask &= weights.notna()
    if not mask.any():
        return None
    values = series[mask]
    if weights is None:
        return float(values.mean())
    weight_slice = weights[mask]
    total = float(weight_slice.sum())
    if not total:
        return None
    return float((values * weight_slice).sum() / total)


def _summarize_forecaster(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    weight_col = df["observations"] if "observations" in df.columns else None
    metrics = {
        "Assets MAE": "assets_mae_mean",
        "Equity MAE": "equity_mae_mean",
        "Net income MAE": "net_income_mae_mean",
        "Identity gap": "identity_gap_mean",
    }
    for label, column in metrics.items():
        if column in df.columns:
            value = _weighted_average(df[column], weight_col)
            lines.append(f"- {label}: {_format_float(value)}")
    coverage_cols = [col for col in df.columns if col.endswith("_interval_coverage")]
    for column in coverage_cols:
        metric = column.replace("_interval_coverage", " interval coverage").replace("_", " ")
        value = _weighted_average(df[column], weight_col)
        lines.append(f"- {metric.title()}: {_format_percent(value)}")
    if "ticker" in df.columns and "assets_mae_mean" in df.columns:
        ticker_stats: list[tuple[str, float]] = []
        for ticker, group in df.groupby("ticker", dropna=False):
            avg = _weighted_average(group["assets_mae_mean"], group.get("observations"))
            if avg is not None:
                ticker_stats.append((str(ticker), avg))
        if ticker_stats:
            best = min(ticker_stats, key=lambda item: item[1])
            worst = max(ticker_stats, key=lambda item: item[1])
            lines.append(
                f"- Lowest assets MAE: {best[0]} ({_format_float(best[1])})"
            )
            lines.append(
                f"- Highest assets MAE: {worst[0]} ({_format_float(worst[1])})"
            )
    return lines or ["- No evaluation metrics available in the summary file."]


def _summarize_scenarios(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    columns = {
        "total_assets_mae": "Assets MAE",
        "total_assets_mape": "Assets MAPE",
        "equity_mae": "Equity MAE",
        "equity_mape": "Equity MAPE",
        "net_income_mae": "Net income MAE",
        "net_income_mape": "Net income MAPE",
        "identity_gap_mae": "Identity-gap MAE",
        "total_assets_interval_coverage": "Assets interval coverage",
        "equity_interval_coverage": "Equity interval coverage",
        "net_income_interval_coverage": "Net income interval coverage",
    }
    display_cols = [col for col in columns if col in df.columns]
    if "scenario" in df.columns and display_cols:
        ordered = df.sort_values("scenario")
        for _, row in ordered.iterrows():
            scenario = row.get("scenario", "unknown")
            parts = [f"**{scenario}**"]
            for column in display_cols:
                label = columns[column]
                value = row.get(column)
                if column.endswith("coverage"):
                    formatted = _format_percent(value)
                else:
                    formatted = _format_float(value)
                parts.append(f"{label}: {formatted}")
            if "observations" in row:
                parts.append(f"n={int(row['observations'])}")
            lines.append("- " + "; ".join(parts))
    else:
        lines.append("- Scenario diagnostics not available.")
    return lines


def _summarize_calibration(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    abs_error_cols = [col for col in df.columns if col.endswith("_abs_error")]
    weight_col = df["observations"] if "observations" in df.columns else None
    if not abs_error_cols:
        return ["- Calibration metrics not available."]
    for column in sorted(abs_error_cols):
        label = column.replace("_", " ")
        value = _weighted_average(df[column], weight_col)
        lines.append(f"- {label.title()}: {_format_float(value)}")
    return lines


def _summarize_macro(df: pd.DataFrame) -> list[str]:
    if "scenario" not in df.columns:
        return ["- Macro-conditioned scenarios not available."]
    lines: list[str] = []
    iterator = (
        df.drop_duplicates(subset=["scenario"]).sort_values("scenario").iterrows()
    )
    for _, row in iterator:
        scenario = row.get("scenario", "unknown")
        macro_payload = row.get("macro_assumptions_json")
        adjustments = row.get("applied_adjustments_json")
        macro_desc = ""
        if isinstance(macro_payload, str) and macro_payload:
            try:
                macro = json.loads(macro_payload)
                if isinstance(macro, dict):
                    macro_desc = ", ".join(
                        f"{key}={value}" for key, value in macro.items()
                    )
            except json.JSONDecodeError:
                macro_desc = macro_payload
        adjustment_count = row.get("applied_adjustment_count")
        adjustment_desc = ""
        if isinstance(adjustments, str) and adjustments:
            try:
                adj = json.loads(adjustments)
                if isinstance(adj, dict):
                    adjustment_desc = ", ".join(sorted(adj))
            except json.JSONDecodeError:
                adjustment_desc = adjustments
        parts = [f"**{scenario}**"]
        if macro_desc:
            parts.append(f"macro: {macro_desc}")
        if adjustment_count is not None and not pd.isna(adjustment_count):
            if adjustment_desc:
                parts.append(
                    f"adjustments: {int(adjustment_count)} ({adjustment_desc})"
                )
            else:
                parts.append(f"adjustments: {int(adjustment_count)}")
        lines.append("- " + "; ".join(parts))
    return lines or ["- Macro-conditioned scenarios not available."]


def _summarize_llm(df: pd.DataFrame) -> list[str]:
    required = {"adapter", "model", "records"}
    if not required.issubset(df.columns):
        return ["- LLM benchmark summary not available."]
    lines: list[str] = []
    for _, row in df.iterrows():
        adapter = row.get("adapter", "unknown")
        model = row.get("model", "default")
        mae = _format_float(row.get("mae_mean")) if "mae_mean" in df.columns else "N/A"
        mae_std = (
            _format_float(row.get("mae_std")) if "mae_std" in df.columns else "N/A"
        )
        coverage = (
            _format_percent(row.get("coverage_mean"))
            if "coverage_mean" in df.columns
            else "N/A"
        )
        coverage_std = (
            _format_percent(row.get("coverage_std"))
            if "coverage_std" in df.columns
            else "N/A"
        )
        seed_count = int(row.get("seed_count", 0)) if "seed_count" in df.columns else 0
        records = int(row.get("records", 0))
        lines.append(
            "- "
            + f"{adapter}/{model}: MAE {mae} (± {mae_std}); coverage {coverage} (± {coverage_std}); "
            + f"records={records}; seeds={seed_count}"
        )
    return lines or ["- LLM benchmark summary not available."]


def _summarize_loan_pricing(payload: dict | None) -> list[str]:
    if not payload:
        return ["- Loan pricing summary not available."]
    if "summary" in payload and isinstance(payload["summary"], list):
        payload = {
            entry.get("scenario", f"scenario_{idx}"): entry
            for idx, entry in enumerate(payload["summary"], start=1)
            if isinstance(entry, dict)
        }
    lines: list[str] = []
    for scenario, info in sorted(payload.items()):
        if not isinstance(info, dict):
            continue
        rate = _format_percent(info.get("avg_rate"))
        spread = _format_percent(info.get("avg_spread"))
        count = info.get("count")
        parts = [f"**{scenario}**"]
        parts.append(f"avg rate: {rate}")
        parts.append(f"avg spread: {spread}")
        if count is not None:
            parts.append(f"n={int(count)}")
        lines.append("- " + "; ".join(parts))
    return lines or ["- Loan pricing summary not available."]


def _summarize_credit(metadata: dict | None) -> list[str]:
    if not metadata:
        return ["- Credit dataset metadata not available."]
    tickers = metadata.get("tickers")
    period_start = metadata.get("period_start")
    period_end = metadata.get("period_end")
    rows = metadata.get("rows")
    ticker_list: list[str] = []
    if isinstance(tickers, Iterable) and not isinstance(tickers, (str, bytes)):
        ticker_list = sorted(str(item) for item in tickers)
    lines = [
        "- Tickers covered: " + (", ".join(ticker_list) if ticker_list else "N/A"),
        f"- Period range: {period_start} → {period_end}",
        f"- Observations: {rows}",
    ]
    return lines


def _summarize_risks(payload: dict | None) -> list[str]:
    summary = None
    if payload and isinstance(payload.get("summary"), list):
        summary = payload["summary"]
    if not summary:
        return ["- Risk warning summary not available."]
    lines: list[str] = []
    grouped: dict[str, int] = {}
    for entry in summary:
        if not isinstance(entry, dict):
            continue
        issuer = entry.get("issuer", "unknown")
        grouped[issuer] = grouped.get(issuer, 0) + int(entry.get("warning_count", 0))
    for issuer, count in sorted(grouped.items(), key=lambda item: item[0]):
        lines.append(f"- {issuer}: {count} flagged warnings")
    return lines


def build_summary(args: argparse.Namespace) -> str:
    sections: list[tuple[str, list[str]]] = []

    forecaster_df = _load_parquet(args.forecaster_summary)
    if forecaster_df is not None:
        sections.append(("Forecast quality", _summarize_forecaster(forecaster_df)))
    else:
        sections.append(("Forecast quality", ["- Evaluation summary not found."]))

    scenario_df = _load_parquet(args.scenario_summary)
    if scenario_df is not None:
        sections.append(("Scenario diagnostics", _summarize_scenarios(scenario_df)))
    else:
        sections.append(("Scenario diagnostics", ["- Scenario summary not found."]))

    calibration_df = _load_parquet(args.calibration_report)
    if calibration_df is not None and not calibration_df.empty:
        sections.append(("Calibration", _summarize_calibration(calibration_df)))
    else:
        sections.append(("Calibration", ["- Calibration report not found or empty."]))

    macro_df = _load_parquet(args.macro_scenarios)
    if macro_df is not None and not macro_df.empty:
        sections.append(("Macro overlays", _summarize_macro(macro_df)))
    else:
        sections.append(("Macro overlays", ["- Macro-conditioned scenarios not found."]))

    llm_df = _load_parquet(args.llm_seed_summary)
    if llm_df is not None and not llm_df.empty:
        sections.append(("LLM benchmarking", _summarize_llm(llm_df)))
    else:
        sections.append(("LLM benchmarking", ["- LLM summary not found."]))

    loan_payload = _load_json(args.loan_pricing_summary)
    sections.append(("Loan pricing", _summarize_loan_pricing(loan_payload)))

    credit_payload = _load_json(args.credit_metadata)
    sections.append(("Credit analytics", _summarize_credit(credit_payload)))

    risk_payload = _load_json(args.risk_summary)
    sections.append(("Risk warnings", _summarize_risks(risk_payload)))

    lines = [f"# {args.title}", ""]
    for title, items in sections:
        lines.append(f"## {title}")
        lines.extend(items)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    summary = build_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    logging.info("Wrote executive summary to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
