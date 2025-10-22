"""Compute coverage diagnostics for probabilistic forecaster outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/forecaster_evaluation.parquet",
        help="Detailed evaluation parquet produced by evaluate_forecaster",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/forecaster_calibration.parquet",
        help="Destination parquet with coverage diagnostics",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="ticker,mode",
        help="Comma separated columns to group by when evaluating calibration",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="assets,equity,net_income",
        help=(
            "Comma separated list of metrics to analyse (subset of assets, "
            "equity, net_income, identity)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level to use for status messages",
    )
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "evaluation": Path,
            "output": Path,
        },
    )


def _parse_tokens(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _quantile_columns(columns: Iterable[str], prefix: str) -> Mapping[float, str]:
    quantiles: dict[float, str] = {}
    for column in columns:
        if not column.startswith(prefix):
            continue
        suffix = column[len(prefix) :]
        if not suffix:
            continue
        try:
            quantile = float(suffix) / 100.0
        except ValueError:
            continue
        quantiles[quantile] = column
    return dict(sorted(quantiles.items()))


def _calibration_metrics(
    frame: pd.DataFrame,
    true_col: str,
    quantile_col: str,
    expected_quantile: float,
) -> tuple[float, float, float]:
    mask = frame[[true_col, quantile_col]].notna().all(axis=1)
    if not mask.any():
        return (float("nan"), float("nan"), float("nan"))
    subset = frame.loc[mask]
    indicator = subset[true_col] <= subset[quantile_col]
    coverage = float(np.mean(indicator.to_numpy(dtype=float)))
    error = coverage - expected_quantile
    return coverage, error, abs(error)


def compute_calibration(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    available_metrics = {
        "assets": ("true_total_assets", "pred_total_assets_q"),
        "equity": ("true_equity", "pred_equity_q"),
        "net_income": ("true_net_income", "pred_net_income_q"),
        "identity": ("identity_gap", "identity_gap_q"),
    }

    selected_metrics = []
    for metric in metrics:
        metric = metric.lower()
        if metric not in available_metrics:
            raise ValueError(
                "Unsupported metric for calibration: " + metric
            )
        true_col, prefix = available_metrics[metric]
        quantile_map = _quantile_columns(df.columns, prefix)
        if not quantile_map:
            logging.debug("Skipping metric %s because no quantiles are present", metric)
            continue
        selected_metrics.append((metric, true_col, quantile_map))

    if not selected_metrics:
        return pd.DataFrame()

    group_cols = list(group_cols)
    frames: list[pd.DataFrame] = []

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
        for keys, frame in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            record = {col: key for col, key in zip(group_cols, keys)}
            record["observations"] = len(frame)
            for metric_name, true_col, quantile_map in selected_metrics:
                for quantile, column in quantile_map.items():
                    coverage, error, abs_error = _calibration_metrics(
                        frame, true_col, column, quantile
                    )
                    suffix = f"{metric_name}_q{int(round(quantile * 100)):02d}"
                    record[f"{suffix}_coverage"] = coverage
                    record[f"{suffix}_error"] = error
                    record[f"{suffix}_abs_error"] = abs_error
            frames.append(pd.DataFrame([record]))
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    record: dict[str, object] = {"observations": len(df)}
    for metric_name, true_col, quantile_map in selected_metrics:
        for quantile, column in quantile_map.items():
            coverage, error, abs_error = _calibration_metrics(
                df, true_col, column, quantile
            )
            suffix = f"{metric_name}_q{int(round(quantile * 100)):02d}"
            record[f"{suffix}_coverage"] = coverage
            record[f"{suffix}_error"] = error
            record[f"{suffix}_abs_error"] = abs_error
    return pd.DataFrame([record])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    group_cols = _parse_tokens(args.group_by)
    metrics = _parse_tokens(args.metrics)

    df = pd.read_parquet(args.evaluation)
    calibration = compute_calibration(df, group_cols, metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_parquet(args.output, index=False)
    logging.info("Calibration diagnostics saved to %s", args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

