"""Validate driver dataset coverage, monotonicity, and feature completeness."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from mlcoe_q1.utils.driver_features import BASE_FEATURES

REQUIRED_FEATURES = list(BASE_FEATURES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drivers",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/driver_features.parquet",
        help="Path to the driver feature parquet file",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=3,
        help="Minimum number of periods required per ticker",
    )
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=500,
        help="Threshold in days before a period gap is flagged",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to persist the validation summary (csv/json/parquet)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    return parser.parse_args(argv)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Driver dataset is missing required columns: {missing}")


def _summarize_group(
    ticker: str,
    group: pd.DataFrame,
    required_columns: Sequence[str],
    min_observations: int,
    max_gap_days: int,
) -> dict:
    ordered = group.sort_values("period").copy()
    ordered["period"] = pd.to_datetime(ordered["period"], errors="coerce")

    duplicates = ordered["period"].duplicated().any()
    invalid_periods = ordered["period"].isna().sum()
    ordered = ordered.dropna(subset=["period"])  # drop invalid timestamps for interval checks

    periods = ordered["period"].to_numpy()
    observation_count = len(ordered)
    insufficient = observation_count < min_observations

    if observation_count >= 2:
        deltas = np.diff(periods).astype("timedelta64[D]").astype(float)
        median_gap = float(np.median(deltas))
        max_gap = float(np.max(deltas))
    else:
        median_gap = float("nan")
        max_gap = float("nan")

    long_gap = bool(np.isfinite(max_gap) and max_gap > max_gap_days)

    na_rows = (
        ordered[required_columns]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .any(axis=1)
        .sum()
    )

    sales_median = float(
        ordered["sales"].apply(pd.to_numeric, errors="coerce").abs().median()
    ) if "sales" in ordered.columns and not ordered["sales"].dropna().empty else float("nan")
    sales_log10 = float(np.log10(sales_median)) if sales_median > 0 else float("nan")

    status = "ok"
    if duplicates or insufficient or na_rows or invalid_periods or long_gap:
        status = "needs_attention"

    return {
        "ticker": ticker,
        "observations": observation_count,
        "min_period": ordered["period"].min(),
        "max_period": ordered["period"].max(),
        "duplicate_periods": bool(duplicates),
        "invalid_periods": int(invalid_periods),
        "na_rows": int(na_rows),
        "insufficient_observations": bool(insufficient),
        "median_gap_days": median_gap,
        "max_gap_days": max_gap,
        "long_gap": long_gap,
        "median_sales_log10": sales_log10,
        "status": status,
    }


def summarise_drivers(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    min_observations: int,
    max_gap_days: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker, group in df.groupby("ticker"):
        rows.append(
            _summarize_group(
                ticker,
                group,
                required_columns,
                min_observations,
                max_gap_days,
            )
        )
    if not rows:
        return pd.DataFrame(columns=[
            "ticker",
            "observations",
            "min_period",
            "max_period",
            "duplicate_periods",
            "invalid_periods",
            "na_rows",
            "insufficient_observations",
            "median_gap_days",
            "max_gap_days",
            "long_gap",
            "median_sales_log10",
            "status",
        ])
    summary = pd.DataFrame(rows)
    return summary.sort_values("ticker").reset_index(drop=True)


def _write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".json"}:
        df.to_json(path, orient="records", indent=2)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format for {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.drivers.exists():
        raise FileNotFoundError(f"Driver dataset not found: {args.drivers}")

    df = pd.read_parquet(args.drivers)
    if df.empty:
        logging.warning("Driver dataset is empty; nothing to validate")
        if args.output:
            _write_output(df, args.output)
        return

    _validate_columns(df, REQUIRED_FEATURES + ["ticker", "period"])

    summary = summarise_drivers(df, REQUIRED_FEATURES, args.min_observations, args.max_gap_days)

    if summary.empty:
        logging.warning("No tickers found after validation")
    else:
        try:
            rendered = summary.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            rendered = summary.to_string(index=False)
        logging.info("Validation summary:\n%s", rendered)

    if args.output:
        _write_output(summary, args.output)
        logging.info("Summary written to %s", args.output)

    issues = summary[summary["status"] != "ok"]
    if not issues.empty:
        logging.error(
            "Validation flagged %d ticker(s) requiring attention: %s",
            len(issues),
            ", ".join(issues["ticker"].tolist()),
        )
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
