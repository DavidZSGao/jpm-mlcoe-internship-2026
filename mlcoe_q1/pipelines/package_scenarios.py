"""Package Monte Carlo forecast outputs into lender-ready scenario tables."""

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
        / "reports/q1/artifacts/forecaster_scenarios.parquet",
        help="Destination parquet containing scenarioised forecasts",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="baseline:0.5,downside:0.1,upside:0.9",
        help=(
            "Comma separated list of scenario_name:quantile pairs. "
            "Use an empty quantile (baseline:) to fall back to the point estimate"
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


def _parse_scenarios(raw: str) -> Mapping[str, float | None]:
    scenarios: dict[str, float | None] = {}
    if not raw.strip():
        return {"baseline": None}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "Scenario specification must be name:quantile; missing ':' in "
                f"{token!r}"
            )
        name, value = token.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError("Scenario name cannot be empty")
        value = value.strip()
        if not value:
            scenarios[name] = None
            continue
        try:
            quantile = float(value)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Invalid quantile for scenario {name}: {value}") from exc
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(
                f"Scenario quantiles must be in [0, 1]; received {quantile} for {name}"
            )
        scenarios[name] = quantile
    if not scenarios:
        scenarios["baseline"] = None
    return scenarios


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


def _select_quantile_column(
    quantile_map: Mapping[float, str], target: float
) -> tuple[float, str] | None:
    if not quantile_map:
        return None
    if target in quantile_map:
        return target, quantile_map[target]
    closest = min(quantile_map, key=lambda q: abs(q - target))
    return closest, quantile_map[closest]


def _value_from_quantile(
    record: Mapping[str, object],
    quantile_map: Mapping[float, str],
    target_quantile: float | None,
    point_key: str,
) -> tuple[float | None, float | None, str]:
    if target_quantile is None:
        value = record.get(point_key)
        return (float(value) if value is not None else None, None, "point_estimate")
    selection = _select_quantile_column(quantile_map, target_quantile)
    if selection is None:
        value = record.get(point_key)
        return (
            float(value) if value is not None else None,
            None,
            "point_estimate",
        )
    used_quantile, column = selection
    value = record.get(column)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        fallback = record.get(point_key)
        return (
            float(fallback) if fallback is not None else None,
            None,
            "point_estimate",
        )
    return float(value), float(used_quantile), "quantile"


def build_scenarios(
    df: pd.DataFrame,
    scenario_map: Mapping[str, float | None],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    scenario_map = dict(scenario_map)
    if not scenario_map:
        scenario_map = {"baseline": None}

    quantiles_assets = _quantile_columns(df.columns, "pred_total_assets_q")
    quantiles_equity = _quantile_columns(df.columns, "pred_equity_q")
    quantiles_income = _quantile_columns(df.columns, "pred_net_income_q")
    quantiles_identity = _quantile_columns(df.columns, "identity_gap_q")

    records: list[dict[str, object]] = []
    base_columns = [
        "ticker",
        "prev_period",
        "target_period",
        "horizon",
        "mode",
        "distribution",
        "mc_strategy",
        "mc_sample_count",
        "true_total_assets",
        "true_equity",
        "true_net_income",
    ]

    for row in df.to_dict(orient="records"):
        for scenario_name, quantile in scenario_map.items():
            scenario_record = {col: row.get(col) for col in base_columns if col in row}
            value_assets, used_q_assets, source_assets = _value_from_quantile(
                row, quantiles_assets, quantile, "pred_total_assets"
            )
            value_equity, used_q_equity, source_equity = _value_from_quantile(
                row, quantiles_equity, quantile, "pred_equity"
            )
            value_income, used_q_income, source_income = _value_from_quantile(
                row, quantiles_income, quantile, "pred_net_income"
            )
            value_identity, used_q_identity, source_identity = _value_from_quantile(
                row, quantiles_identity, quantile, "identity_gap"
            )

            scenario_record.update(
                {
                    "scenario": scenario_name,
                    "scenario_quantile": used_q_assets
                    if used_q_assets is not None
                    else quantile,
                    "pred_total_assets": value_assets,
                    "pred_equity": value_equity,
                    "pred_net_income": value_income,
                    "identity_gap": value_identity,
                    "scenario_source_assets": source_assets,
                    "scenario_source_equity": source_equity,
                    "scenario_source_net_income": source_income,
                    "scenario_source_identity": source_identity,
                }
            )

            records.append(scenario_record)

    return pd.DataFrame(records)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    scenarios = _parse_scenarios(args.scenarios)
    df = pd.read_parquet(args.evaluation)

    packaged = build_scenarios(df, scenarios)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    packaged.to_parquet(args.output, index=False)
    logging.info("Scenario table saved to %s", args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

