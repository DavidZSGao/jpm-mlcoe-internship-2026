"""Assess scenario tables against realised filings to gauge reasonableness."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/forecaster_scenarios.parquet",
        help="Scenario table produced by package_scenarios",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/scenario_reasonableness.parquet",
        help="Destination parquet for aggregated diagnostics",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="scenario",
        help=(
            "Comma separated list of columns to group by when summarising "
            "scenario accuracy (default: scenario). Use an empty string "
            "for an overall summary."
        ),
    )
    parser.add_argument(
        "--interval-lower",
        type=str,
        default=None,
        help="Scenario name representing the downside/bottom interval bound",
    )
    parser.add_argument(
        "--interval-upper",
        type=str,
        default=None,
        help="Scenario name representing the upside/top interval bound",
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
            "scenarios": Path,
            "output": Path,
        },
    )


def _safe_mape(pred: np.ndarray, actual: np.ndarray) -> float | None:
    mask = (~np.isnan(pred)) & (~np.isnan(actual)) & (actual != 0)
    if not mask.any():
        return None
    errors = np.abs((pred[mask] - actual[mask]) / actual[mask])
    return float(errors.mean())


def compute_scenario_statistics(
    df: pd.DataFrame, group_cols: Sequence[str]
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    helper_column: str | None = None
    parsed_group_cols = [col for col in group_cols if col]
    if not parsed_group_cols:
        helper_column = "__all__"
        df[helper_column] = "all"
        parsed_group_cols = [helper_column]

    records: list[dict[str, object]] = []
    for keys, group in df.groupby(parsed_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {col: value for col, value in zip(parsed_group_cols, keys)}
        record["observations"] = int(len(group))

        for metric_name, pred_key, actual_key in [
            ("total_assets", "pred_total_assets", "true_total_assets"),
            ("equity", "pred_equity", "true_equity"),
            ("net_income", "pred_net_income", "true_net_income"),
        ]:
            pred = pd.to_numeric(group.get(pred_key), errors="coerce")
            actual = pd.to_numeric(group.get(actual_key), errors="coerce")
            mask = pred.notna() & actual.notna()
            if mask.any():
                diff = pred[mask] - actual[mask]
                record[f"{metric_name}_mae"] = float(np.abs(diff).mean())
                record[f"{metric_name}_bias"] = float(diff.mean())
                mape = _safe_mape(pred[mask].to_numpy(), actual[mask].to_numpy())
                record[f"{metric_name}_mape"] = mape
            else:
                record[f"{metric_name}_mae"] = None
                record[f"{metric_name}_bias"] = None
                record[f"{metric_name}_mape"] = None

        identity_values = pd.to_numeric(group.get("identity_gap"), errors="coerce")
        identity_values = identity_values[identity_values.notna()]
        if not identity_values.empty:
            record["identity_gap_mae"] = float(np.abs(identity_values).mean())
            record["identity_gap_bias"] = float(identity_values.mean())
        else:
            record["identity_gap_mae"] = None
            record["identity_gap_bias"] = None

        quantile_flags = (
            (group.get("scenario_source_assets") == "quantile")
            | (group.get("scenario_source_equity") == "quantile")
            | (group.get("scenario_source_net_income") == "quantile")
        )
        record["quantile_rows"] = int(quantile_flags.fillna(False).sum())
        records.append(record)

    result = pd.DataFrame.from_records(records)
    if parsed_group_cols:
        result = result.sort_values(parsed_group_cols).reset_index(drop=True)
    if helper_column and helper_column in result.columns:
        result = result.drop(columns=[helper_column])
    return result


def _record_key_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "ticker",
        "prev_period",
        "target_period",
        "horizon",
        "mode",
        "distribution",
        "mc_strategy",
    ]
    return [col for col in preferred if col in df.columns]


def _build_interval_records(
    df: pd.DataFrame,
    lower: str,
    upper: str,
    grouping_columns: Sequence[str],
) -> pd.DataFrame:
    if lower is None or upper is None:
        return pd.DataFrame()
    if lower == upper:
        raise ValueError("Lower and upper scenarios must differ for interval coverage")

    required = {lower, upper}
    subset = df[df["scenario"].isin(required)]
    if subset.empty:
        return pd.DataFrame()

    key_cols = _record_key_columns(subset)
    if not key_cols:
        subset = subset.copy()
        subset["__row__"] = np.arange(len(subset))
        key_cols = ["__row__"]

    interval_records: list[dict[str, object]] = []
    for _, record_df in subset.groupby(key_cols, dropna=False):
        scenarios_present = set(record_df["scenario"].unique())
        if not required.issubset(scenarios_present):
            continue
        lower_row = record_df[record_df["scenario"] == lower].iloc[0]
        upper_row = record_df[record_df["scenario"] == upper].iloc[0]
        base_row = record_df.iloc[0]

        entry = {col: base_row.get(col) for col in grouping_columns if col in record_df}
        entry["interval_observations"] = 1

        for metric, pred_key in [
            ("total_assets", "pred_total_assets"),
            ("equity", "pred_equity"),
            ("net_income", "pred_net_income"),
        ]:
            true_key = f"true_{metric}"
            actual = pd.to_numeric(base_row.get(true_key), errors="coerce")
            lower_pred = pd.to_numeric(lower_row.get(pred_key), errors="coerce")
            upper_pred = pd.to_numeric(upper_row.get(pred_key), errors="coerce")
            width = None
            covered = None
            if lower_pred is not None and upper_pred is not None:
                if np.isnan(lower_pred) or np.isnan(upper_pred):
                    width = None
                else:
                    width = float(upper_pred - lower_pred)
                if actual is not None and not np.isnan(actual):
                    if width is None:
                        covered = None
                    else:
                        lower_bound = float(lower_pred)
                        upper_bound = float(upper_pred)
                        if lower_bound > upper_bound:
                            lower_bound, upper_bound = upper_bound, lower_bound
                        covered = float(lower_bound <= float(actual) <= upper_bound)
            entry[f"{metric}_interval_width"] = width
            entry[f"{metric}_interval_covered"] = covered

        interval_records.append(entry)

    if not interval_records:
        return pd.DataFrame()

    interval_df = pd.DataFrame(interval_records)
    group_cols = [col for col in grouping_columns if col in interval_df.columns]
    if not group_cols:
        interval_df["__all__"] = "all"
        group_cols = ["__all__"]

    agg: dict[str, str] = {"interval_observations": "sum"}
    for metric in ["total_assets", "equity", "net_income"]:
        agg[f"{metric}_interval_covered"] = "mean"
        agg[f"{metric}_interval_width"] = "mean"

    summary = interval_df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    summary = summary.rename(
        columns={
            f"{metric}_interval_covered": f"{metric}_interval_coverage"
            for metric in ["total_assets", "equity", "net_income"]
        }
    )
    return summary


def evaluate_scenarios(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    lower: str | None,
    upper: str | None,
) -> pd.DataFrame:
    stats = compute_scenario_statistics(df, group_cols)
    if stats.empty:
        return stats

    merge_columns = [col for col in group_cols if col and col != "scenario"]
    helper_column = None
    if not merge_columns:
        helper_column = "__all__"
        stats[helper_column] = "all"
        merge_columns = [helper_column]

    coverage = _build_interval_records(df, lower, upper, merge_columns)
    if not coverage.empty:
        stats = stats.merge(coverage, on=merge_columns, how="left")
    if helper_column and helper_column in stats.columns:
        stats = stats.drop(columns=[helper_column])

    return stats


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    logging.info("Loading scenarios from %s", args.scenarios)
    scenario_df = pd.read_parquet(args.scenarios)

    group_cols = [col.strip() for col in args.group_by.split(",") if col.strip()]
    summary = evaluate_scenarios(scenario_df, group_cols, args.interval_lower, args.interval_upper)

    logging.info("Writing scenario diagnostics to %s", args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(args.output, index=False)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
