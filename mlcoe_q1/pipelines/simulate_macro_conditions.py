"""Apply macro-conditioned shocks to packaged balance-sheet scenarios."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build CLI arguments for the macro-conditioned simulator."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=None,
        help=(
            "Optional path to a JSON file describing macro scenarios. "
            "Falls back to the built-in baseline/downside/ratchet set when omitted."
        ),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/forecaster_scenarios.parquet",
        help="Scenario parquet produced by package_scenarios or evaluate_forecaster",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/macro_conditioned_scenarios.parquet",
        help="Destination parquet containing macro-conditioned scenarios",
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
            "evaluation": Path,
            "output": Path,
        },
    )


def default_macro_scenarios() -> list[dict[str, Any]]:
    """Return a conservative baseline of macro-conditioned scenario definitions."""

    return [
        {
            "name": "baseline",
            "description": "Point forecast aligned with consensus macro assumptions",
            "macro": {
                "gdp_growth": 0.018,
                "unemployment_rate": 0.037,
                "policy_rate": 0.0425,
            },
            "adjustments": {},
            "source": "default",
        },
        {
            "name": "mild_downturn",
            "description": (
                "GDP slows 150 bps with 120 bps unemployment uptick; margins and equity tighten"
            ),
            "macro": {
                "gdp_growth": 0.003,
                "unemployment_rate": 0.049,
                "policy_rate": 0.038,
            },
            "adjustments": {
                "pred_net_income": {"mode": "pct", "value": -0.08},
                "pred_equity": {"mode": "pct", "value": -0.0125},
                "identity_gap": {"mode": "add", "value": 0.0},
            },
            "source": "default",
        },
        {
            "name": "rate_shock",
            "description": (
                "Policy rate jumps 200 bps; asset growth cools and payout is curbed to preserve equity"
            ),
            "macro": {
                "gdp_growth": 0.01,
                "unemployment_rate": 0.042,
                "policy_rate": 0.0625,
            },
            "adjustments": {
                "pred_total_assets": {"mode": "pct", "value": -0.015},
                "pred_net_income": {"mode": "pct", "value": -0.05},
                "pred_equity": {"mode": "pct", "value": -0.02},
            },
            "source": "default",
        },
    ]


def _normalise_scenario(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = raw.get("name")
    if not required:
        raise ValueError("Scenario definition must include a non-empty 'name'")
    macro = raw.get("macro") or {}
    if not isinstance(macro, Mapping):
        raise ValueError("Scenario 'macro' field must be a mapping of indicators to values")
    adjustments = raw.get("adjustments") or {}
    if not isinstance(adjustments, Mapping):
        raise ValueError("Scenario 'adjustments' field must be a mapping of metric shocks")
    description = raw.get("description") or ""
    source = raw.get("source") or "config"
    return {
        "name": str(required),
        "description": str(description),
        "macro": dict(macro),
        "adjustments": {
            str(metric): dict(spec) for metric, spec in adjustments.items()
        },
        "source": str(source),
    }


def load_macro_scenarios(path: Path | None) -> list[dict[str, Any]]:
    """Load macro scenario definitions from disk or fall back to defaults."""

    if path is None:
        return [
            _normalise_scenario(scenario)
            for scenario in default_macro_scenarios()
        ]

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, Mapping) and "scenarios" in raw:
        candidates = raw["scenarios"]
    else:
        candidates = raw

    if not isinstance(candidates, Sequence):
        raise ValueError("Scenario config must be a list or contain a 'scenarios' list")

    return [_normalise_scenario(candidate) for candidate in candidates]


def _apply_adjustment(value: Any, spec: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    if value is None:
        return value, {}

    if isinstance(value, (float, int)):
        base_value = float(value)
    else:
        try:
            base_value = float(value)
        except (TypeError, ValueError):
            return value, {}

    if not np.isfinite(base_value):
        return value, {}

    mode = str(spec.get("mode", "pct")).lower()
    amount = spec.get("value", 0.0)
    try:
        delta = float(amount)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Adjustment value must be numeric; received {amount!r}") from exc

    if mode in {"pct", "percent", "percentage"}:
        adjusted = base_value * (1.0 + delta)
    elif mode in {"add", "delta", "absolute"}:
        adjusted = base_value + delta
    elif mode in {"set", "override"}:
        adjusted = delta
    else:
        raise ValueError(f"Unsupported adjustment mode {mode!r} for macro scenario")

    return adjusted, {
        "mode": mode,
        "value": delta,
        "base": base_value,
        "adjusted": adjusted,
    }


def apply_macro_scenarios(
    df: pd.DataFrame, scenarios: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    """Apply macro-driven adjustments to each row in the scenario table."""

    if df.empty:
        return pd.DataFrame()

    records: list[MutableMapping[str, Any]] = []
    base_columns = list(df.columns)
    for row in df.to_dict(orient="records"):
        for scenario in scenarios:
            scenario_record: MutableMapping[str, Any] = {col: row.get(col) for col in base_columns}
            applied: dict[str, Any] = {}
            for metric, spec in scenario.get("adjustments", {}).items():
                if metric not in scenario_record:
                    continue
                adjusted, metadata = _apply_adjustment(scenario_record[metric], spec)
                if metadata:
                    scenario_record[metric] = adjusted
                    applied[metric] = metadata
            macro = scenario.get("macro", {})
            for indicator, value in macro.items():
                scenario_record[f"macro_{indicator}"] = value
            scenario_record["scenario"] = scenario.get("name")
            scenario_record["scenario_description"] = scenario.get("description", "")
            scenario_record["scenario_source"] = scenario.get("source", "config")
            scenario_record["macro_assumptions_json"] = json.dumps(
                macro, sort_keys=True
            )
            scenario_record["applied_adjustments_json"] = json.dumps(
                applied, sort_keys=True
            )
            scenario_record["applied_adjustment_count"] = len(applied)
            records.append(scenario_record)

    return pd.DataFrame.from_records(records)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    logger = logging.getLogger("macro_conditions")

    logger.info("Loading evaluation data from %s", args.evaluation)
    df = pd.read_parquet(args.evaluation)
    logger.info("Loaded %d scenario rows", len(df))

    scenarios = load_macro_scenarios(args.scenarios)
    logger.info("Applying %d macro scenarios", len(scenarios))

    conditioned = apply_macro_scenarios(df, scenarios)
    logger.info(
        "Generated %d macro-conditioned rows across %d base observations",
        len(conditioned),
        len(df),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    conditioned.to_parquet(args.output, index=False)
    logger.info("Macro-conditioned scenarios written to %s", args.output)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

