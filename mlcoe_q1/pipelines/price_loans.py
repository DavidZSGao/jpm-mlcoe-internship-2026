"""Price loans by combining scenario projections with credit metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

from mlcoe_q1.credit.loan_pricing import (
    DEFAULT_SPREADS,
    LoanPricingParameters,
    price_scenarios,
)
from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


SCENARIO_REQUIRED_COLUMNS = {
    "ticker",
    "target_period",
    "scenario",
    "pred_total_assets",
    "pred_equity",
    "pred_net_income",
}


def _load_scenarios(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = SCENARIO_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Scenario parquet missing required columns: " + ", ".join(sorted(missing))
        )
    return frame


def _normalise_period(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.tz_localize(None)


def _attach_credit_metadata(
    scenarios: pd.DataFrame,
    credit: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if credit is None or credit.empty:
        scenarios = scenarios.copy()
        scenarios["rating_bucket"] = "unavailable"
        scenarios["z_score"] = pd.NA
        scenarios["leverage"] = pd.NA
        return scenarios

    credit = credit.copy()
    credit["period_ts"] = _normalise_period(credit["period"])

    scenarios = scenarios.copy()
    scenarios["target_period_ts"] = _normalise_period(scenarios["target_period"])

    merged = scenarios.merge(
        credit,
        how="left",
        left_on=["ticker", "target_period_ts"],
        right_on=["ticker", "period_ts"],
        suffixes=("", "_credit"),
    )

    rating_cols = ["rating_bucket", "z_score", "leverage"]
    missing_mask = merged["rating_bucket"].isna()
    if missing_mask.any():
        latest = (
            credit.sort_values("period_ts")
            .dropna(subset=["rating_bucket"])
            .drop_duplicates(subset=["ticker"], keep="last")
            .set_index("ticker")
        )
        for col in rating_cols:
            if col in latest.columns:
                merged.loc[missing_mask, col] = merged.loc[missing_mask, "ticker"].map(
                    latest[col]
                )

    merged["rating_bucket"] = merged["rating_bucket"].fillna("unavailable")
    return merged


def _parse_spreads(payload: Optional[str]) -> Mapping[str, float]:
    if not payload:
        return DEFAULT_SPREADS
    data: Dict[str, float] = json.loads(payload)
    return {key.lower(): float(value) for key, value in data.items()}


def _parse_macro_config(payload: Optional[str]) -> Mapping[str, Mapping[str, float]]:
    if not payload:
        return {}

    candidate_path = Path(payload)
    if candidate_path.exists():
        raw = json.loads(candidate_path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(payload)

    if isinstance(raw, Mapping):
        items = raw.items()
    elif isinstance(raw, Iterable):
        items = ((entry.get("indicator"), entry) for entry in raw if isinstance(entry, Mapping))
    else:
        raise ValueError("Macro sensitivity config must be a mapping or list of mappings")

    config: Dict[str, Dict[str, float]] = {}
    for indicator, spec in items:
        if indicator is None:
            continue
        if not isinstance(spec, Mapping):
            continue
        config[str(indicator)] = dict(spec)
    return config


def _summarise(priced: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    grouped = priced.groupby("scenario")
    macro_columns = [col for col in priced.columns if col.startswith("macro_adj_")]
    summary: Dict[str, Dict[str, float]] = {}
    for scenario, frame in grouped:
        summary[scenario] = {
            "count": int(len(frame)),
            "avg_rate": float(frame["recommended_rate"].mean()),
            "avg_spread": float(
                (frame["recommended_rate"] - frame["base_rate"]).mean()
            ),
            "avg_leverage": float(frame["leverage_ratio"].mean(skipna=True)),
        }
        if "macro_adjustment" in frame.columns:
            summary[scenario]["avg_macro_adjustment"] = float(
                frame["macro_adjustment"].mean()
            )
        for column in macro_columns:
            value = frame[column].mean()
            summary_key = f"avg_{column}"
            summary[scenario][summary_key] = float(value) if pd.notna(value) else 0.0
    return summary


def build_pricing_table(
    scenarios: pd.DataFrame,
    credit: Optional[pd.DataFrame],
    *,
    params: LoanPricingParameters,
    spreads: Mapping[str, float],
    macro_config: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> pd.DataFrame:
    enriched = _attach_credit_metadata(scenarios, credit)
    priced = price_scenarios(
        enriched,
        params=params,
        spreads=spreads,
        macro_config=macro_config,
    )
    columns = [
        "ticker",
        "target_period",
        "scenario",
        "rating_bucket",
        "z_score",
        "leverage",
        "leverage_ratio",
        "net_income_margin",
        "base_rate",
        "base_spread",
        "leverage_adjustment",
        "coverage_adjustment",
        "scenario_adjustment",
        "macro_adjustment",
        "recommended_rate",
        "pricing_notes",
    ]
    existing = [col for col in columns if col in priced.columns]
    return priced[existing + [col for col in priced.columns if col not in existing]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Price loans using scenario forecasts and credit metrics",
    )
    add_config_argument(parser)
    parser.add_argument(
        "scenarios",
        type=Path,
        help="Scenario parquet produced by package_scenarios",
    )
    parser.add_argument(
        "--credit-dataset",
        type=Path,
        help="Optional Altman feature parquet to supply rating buckets and leverage",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=LoanPricingParameters().risk_free_rate,
        help="Risk-free rate assumption (decimal).",
    )
    parser.add_argument(
        "--spread-config",
        help="JSON mapping of rating bucket to spread (decimal)",
    )
    parser.add_argument(
        "--macro-sensitivity",
        help=(
            "JSON string or file path describing macro indicator sensitivities. "
            "For example: {\"macro_policy_rate\": {\"baseline\": \"risk_free_rate\", "
            "\"multiplier\": 0.5}}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional parquet destination for the pricing table",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional JSON summary of average rates by scenario",
    )
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "scenarios": Path,
            "credit_dataset": Path,
            "output": Path,
            "summary_output": Path,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    scenarios = _load_scenarios(args.scenarios)
    credit = pd.read_parquet(args.credit_dataset) if args.credit_dataset else None

    spreads = _parse_spreads(args.spread_config)
    macro_config = _parse_macro_config(args.macro_sensitivity)
    params = LoanPricingParameters(risk_free_rate=args.risk_free_rate)
    priced = build_pricing_table(
        scenarios,
        credit,
        params=params,
        spreads=spreads,
        macro_config=macro_config,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        priced.to_parquet(args.output, index=False)

    summary = _summarise(priced)
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
