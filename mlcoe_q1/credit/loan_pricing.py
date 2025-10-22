"""Loan pricing utilities for Strategic Lending scenarios."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

import math

import json

import numpy as np
import pandas as pd


DEFAULT_SPREADS: Mapping[str, float] = {
    "investment_grade": 0.015,
    "bbb": 0.02,
    "bb": 0.035,
    "b": 0.05,
    "ccc": 0.08,
    "unavailable": 0.06,
}


@dataclass(frozen=True)
class LoanPricingParameters:
    """Configuration for translating risk metrics into loan rates."""

    risk_free_rate: float = 0.045
    leverage_target: float = 0.5
    leverage_factor: float = 0.04
    coverage_target: float = 0.04
    coverage_factor: float = 0.5
    scenario_factor: float = 0.04
    floor_rate: float = 0.02


@dataclass(frozen=True)
class LoanPricingResult:
    """Decomposed loan pricing output for a single borrower scenario."""

    base_rate: float
    base_spread: float
    leverage_adjustment: float
    coverage_adjustment: float
    scenario_adjustment: float
    macro_adjustment: float
    recommended_rate: float
    leverage_ratio: Optional[float]
    net_income_margin: Optional[float]
    notes: str
    macro_breakdown: Dict[str, float]


def normalise_spread_table(spreads: Mapping[str, float]) -> Dict[str, float]:
    """Ensure the spread mapping contains an ``unavailable`` fallback."""

    table: MutableMapping[str, float] = dict(spreads)
    table.setdefault("unavailable", DEFAULT_SPREADS["unavailable"])
    return dict(table)


def _clip_rate(value: float, minimum: float) -> float:
    return float(max(value, minimum))


def compute_pricing(
    *,
    rating_bucket: Optional[str],
    pred_total_assets: Optional[float],
    pred_equity: Optional[float],
    pred_net_income: Optional[float],
    scenario_quantile: Optional[float],
    params: LoanPricingParameters,
    spreads: Mapping[str, float] = DEFAULT_SPREADS,
    macro_values: Optional[Mapping[str, Any]] = None,
    macro_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> LoanPricingResult:
    """Translate scenario metrics into a recommended loan rate."""

    spread_table = normalise_spread_table(spreads)
    bucket = (rating_bucket or "unavailable").lower()
    base_spread = float(spread_table.get(bucket, spread_table["unavailable"]))

    leverage_ratio: Optional[float] = None
    if pred_total_assets and pred_equity is not None and pred_total_assets > 0:
        leverage_ratio = (pred_total_assets - pred_equity) / pred_total_assets
        leverage_ratio = float(np.clip(leverage_ratio, 0.0, 5.0))
    elif pred_total_assets and pred_total_assets > 0:
        leverage_ratio = None

    leverage_adjustment = 0.0
    note_parts: list[str] = []
    if leverage_ratio is not None and leverage_ratio > params.leverage_target:
        leverage_adjustment = (leverage_ratio - params.leverage_target) * params.leverage_factor
        note_parts.append(
            f"Leverage {leverage_ratio:.2f} above {params.leverage_target:.2f}"
        )

    net_income_margin: Optional[float] = None
    coverage_adjustment = 0.0
    if pred_total_assets and pred_net_income is not None and pred_total_assets > 0:
        net_income_margin = pred_net_income / pred_total_assets
        if math.isfinite(net_income_margin):
            coverage_adjustment = (params.coverage_target - net_income_margin) * params.coverage_factor
            coverage_adjustment = float(np.clip(coverage_adjustment, -0.05, 0.1))
            if coverage_adjustment > 0:
                note_parts.append(
                    f"Margin {net_income_margin:.3f} below target {params.coverage_target:.3f}"
                )
            elif coverage_adjustment < 0:
                note_parts.append(
                    f"Margin {net_income_margin:.3f} above target {params.coverage_target:.3f}"
                )
        else:
            net_income_margin = None
            coverage_adjustment = 0.0

    scenario_adjustment = 0.0
    if scenario_quantile is not None and math.isfinite(scenario_quantile):
        scenario_adjustment = (0.5 - scenario_quantile) * params.scenario_factor
        scenario_adjustment = float(np.clip(scenario_adjustment, -0.04, 0.06))
        if scenario_adjustment > 0:
            note_parts.append(
                f"Downside quantile {scenario_quantile:.2f} adds stress buffer"
            )
        elif scenario_adjustment < 0:
            note_parts.append(
                f"Upside quantile {scenario_quantile:.2f} reduces premium"
            )

    macro_adjustment = 0.0
    macro_breakdown: Dict[str, float] = {}
    if macro_values and macro_config:
        for indicator, spec_raw in macro_config.items():
            if spec_raw is None:
                continue
            try:
                raw_value = macro_values.get(indicator)
            except AttributeError:
                continue
            if raw_value is None:
                continue
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue

            spec = dict(spec_raw)
            multiplier = spec.get("multiplier", spec.get("sensitivity", 0.0))
            try:
                multiplier_value = float(multiplier)
            except (TypeError, ValueError):
                continue

            baseline_raw = spec.get("baseline")
            if isinstance(baseline_raw, str) and baseline_raw.lower() == "risk_free_rate":
                baseline_value = params.risk_free_rate
            elif baseline_raw is None:
                baseline_value = 0.0
            else:
                try:
                    baseline_value = float(baseline_raw)
                except (TypeError, ValueError):
                    baseline_value = 0.0

            mode = str(spec.get("mode", "difference")).lower()
            if mode in {"difference", "delta"}:
                delta = numeric_value - baseline_value
            elif mode in {"value", "direct"}:
                delta = numeric_value
            else:
                # Unsupported modes are ignored rather than raising to keep pricing resilient
                continue

            adjustment = delta * multiplier_value
            macro_adjustment += adjustment
            macro_breakdown[indicator] = macro_breakdown.get(indicator, 0.0) + adjustment

            template = spec.get("note")
            if isinstance(template, str):
                try:
                    note_parts.append(
                        template.format(
                            indicator=indicator,
                            value=numeric_value,
                            baseline=baseline_value,
                            delta=delta,
                            adjustment=adjustment,
                            multiplier=multiplier_value,
                        )
                    )
                except Exception:  # pragma: no cover - defensive formatting guard
                    note_parts.append(
                        f"Macro {indicator} {numeric_value:.3f} adds {adjustment:+.3f}"
                    )
            elif adjustment != 0.0:
                note_parts.append(
                    f"Macro {indicator} {numeric_value:.3f} adjusts rate {adjustment:+.3f}"
                )

    base_rate = params.risk_free_rate
    recommended = (
        base_rate
        + base_spread
        + leverage_adjustment
        + coverage_adjustment
        + scenario_adjustment
        + macro_adjustment
    )
    recommended = _clip_rate(recommended, params.floor_rate)

    notes = "; ".join(note_parts)

    return LoanPricingResult(
        base_rate=base_rate,
        base_spread=base_spread,
        leverage_adjustment=leverage_adjustment,
        coverage_adjustment=coverage_adjustment,
        scenario_adjustment=scenario_adjustment,
        macro_adjustment=macro_adjustment,
        recommended_rate=recommended,
        leverage_ratio=leverage_ratio,
        net_income_margin=net_income_margin,
        notes=notes,
        macro_breakdown=dict(macro_breakdown),
    )


def price_scenarios(
    scenarios: pd.DataFrame,
    *,
    params: LoanPricingParameters,
    spreads: Mapping[str, float] = DEFAULT_SPREADS,
    macro_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """Apply loan pricing to each scenario row."""

    scenarios = scenarios.copy()

    macro_columns = [col for col in scenarios.columns if col.startswith("macro_")]
    results = []
    for _, row in scenarios.iterrows():
        macro_values: dict[str, Any] = {}
        for column in macro_columns:
            value = row.get(column)
            macro_values[column] = value
            suffix = column[len("macro_") :]
            macro_values.setdefault(suffix, value)
        pricing = compute_pricing(
            rating_bucket=row.get("rating_bucket"),
            pred_total_assets=row.get("pred_total_assets"),
            pred_equity=row.get("pred_equity"),
            pred_net_income=row.get("pred_net_income"),
            scenario_quantile=row.get("scenario_quantile"),
            params=params,
            spreads=spreads,
            macro_values=macro_values,
            macro_config=macro_config,
        )
        payload = row.to_dict()
        payload.update(
            {
                "base_rate": pricing.base_rate,
                "base_spread": pricing.base_spread,
                "leverage_adjustment": pricing.leverage_adjustment,
                "coverage_adjustment": pricing.coverage_adjustment,
                "scenario_adjustment": pricing.scenario_adjustment,
                "macro_adjustment": pricing.macro_adjustment,
                "recommended_rate": pricing.recommended_rate,
                "leverage_ratio": (
                    float(pricing.leverage_ratio)
                    if pricing.leverage_ratio is not None
                    else np.nan
                ),
                "net_income_margin": (
                    float(pricing.net_income_margin)
                    if pricing.net_income_margin is not None
                    else np.nan
                ),
                "pricing_notes": pricing.notes,
                "macro_breakdown": json.dumps(pricing.macro_breakdown),
            }
        )
        for indicator, adjustment in pricing.macro_breakdown.items():
            payload[f"macro_adj_{indicator}"] = adjustment
        results.append(payload)

    return pd.DataFrame(results)
