"""Lightweight structural templates for projecting bank balance sheets."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping

import numpy as np

from .balance_sheet_constraints import BalanceSheetState, ProjectionResult

ASSET_FIELDS = (
    "cash",
    "receivables",
    "inventory",
    "other_current_assets",
    "net_pp_and_e",
    "other_non_current_assets",
)

LIABILITY_FIELDS = (
    "accounts_payable",
    "short_term_debt",
    "accrued_expenses",
    "long_term_debt",
    "other_liabilities",
)


@dataclass
class BankTemplate:
    """Encapsulates a proportional balance-sheet template for a bank ticker."""

    ticker: str
    asset_growth: float
    liability_ratio: float
    asset_weights: Dict[str, float]
    liability_weights: Dict[str, float]

    def project(self, previous: BalanceSheetState) -> ProjectionResult:
        """Project one step forward using proportional scaling heuristics."""

        prev_assets = previous.total_assets()
        growth_factor = 1.0 + self.asset_growth
        if prev_assets <= 0:
            prev_assets = 1.0
        total_assets = prev_assets * growth_factor

        assets: Dict[str, float] = {}
        for field in ASSET_FIELDS:
            weight = self.asset_weights.get(field, 0.0)
            assets[field] = total_assets * max(weight, 0.0)

        residual_assets = total_assets - sum(assets.values())
        if residual_assets != 0.0:
            assets["cash"] = assets.get("cash", 0.0) + residual_assets

        total_liabilities = total_assets * max(self.liability_ratio, 0.0)
        liabilities: Dict[str, float] = {}
        for field in LIABILITY_FIELDS:
            weight = self.liability_weights.get(field, 0.0)
            liabilities[field] = total_liabilities * max(weight, 0.0)

        residual_liabilities = total_liabilities - sum(liabilities.values())
        if residual_liabilities != 0.0:
            liabilities["short_term_debt"] = liabilities.get("short_term_debt", 0.0) + residual_liabilities

        equity = max(0.0, total_assets - total_liabilities)

        state = BalanceSheetState(
            cash=assets.get("cash", 0.0),
            receivables=assets.get("receivables", 0.0),
            inventory=assets.get("inventory", 0.0),
            other_current_assets=assets.get("other_current_assets", 0.0),
            net_pp_and_e=assets.get("net_pp_and_e", 0.0),
            other_non_current_assets=assets.get("other_non_current_assets", 0.0),
            accounts_payable=liabilities.get("accounts_payable", 0.0),
            short_term_debt=liabilities.get("short_term_debt", 0.0),
            accrued_expenses=liabilities.get("accrued_expenses", 0.0),
            long_term_debt=liabilities.get("long_term_debt", 0.0),
            other_liabilities=liabilities.get("other_liabilities", 0.0),
            equity=equity,
        )

        identity_gap = state.total_assets() - (state.total_liabilities() + state.equity)
        return ProjectionResult(
            state=state,
            income_statement={},
            cash_flow_statement={},
            identity_gap=identity_gap,
        )


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def compute_bank_template(
    ticker: str, states: Mapping[np.datetime64, BalanceSheetState]
) -> BankTemplate:
    """Derive a :class:`BankTemplate` from historical balance sheet states."""

    if not states:
        raise ValueError(f"No states available to build bank template for {ticker}")

    ordered_periods = sorted(states.keys())
    total_assets = np.array([states[p].total_assets() for p in ordered_periods], dtype=float)

    growth_series = []
    for i in range(1, len(total_assets)):
        prev = total_assets[i - 1]
        if prev > 0:
            growth_series.append((total_assets[i] - prev) / prev)
    asset_growth = float(np.median(growth_series)) if growth_series else 0.0

    liability_ratios = []
    asset_weights_acc: Dict[str, list[float]] = {field: [] for field in ASSET_FIELDS}
    liability_weights_acc: Dict[str, list[float]] = {field: [] for field in LIABILITY_FIELDS}

    for period in ordered_periods:
        state = states[period]
        assets_total = state.total_assets()
        liabilities_total = state.total_liabilities()
        if assets_total > 0:
            for field in ASSET_FIELDS:
                asset_weights_acc[field].append(getattr(state, field) / assets_total)
        if liabilities_total > 0:
            liability_ratios.append(liabilities_total / assets_total if assets_total > 0 else 0.0)
            for field in LIABILITY_FIELDS:
                liability_weights_acc[field].append(getattr(state, field) / liabilities_total)

    liability_ratio = float(liability_ratios[-1]) if liability_ratios else 0.9

    asset_weights = {
        field: float(np.mean(values)) if values else 0.0 for field, values in asset_weights_acc.items()
    }
    liability_weights = {
        field: float(np.mean(values)) if values else 0.0
        for field, values in liability_weights_acc.items()
    }

    return BankTemplate(
        ticker=ticker,
        asset_growth=asset_growth,
        liability_ratio=liability_ratio,
        asset_weights=_normalize(asset_weights),
        liability_weights=_normalize(liability_weights),
    )


def serialize_templates(templates: Iterable[BankTemplate]) -> Dict[str, dict]:
    return {template.ticker: asdict(template) for template in templates}


def deserialize_templates(payload: Mapping[str, dict]) -> Dict[str, BankTemplate]:
    out: Dict[str, BankTemplate] = {}
    for ticker, data in payload.items():
        out[ticker.upper()] = BankTemplate(
            ticker=data["ticker"],
            asset_growth=float(data.get("asset_growth", 0.0)),
            liability_ratio=float(data.get("liability_ratio", 0.0)),
            asset_weights=dict(data.get("asset_weights", {})),
            liability_weights=dict(data.get("liability_weights", {})),
        )
    return out


__all__ = [
    "BankTemplate",
    "compute_bank_template",
    "serialize_templates",
    "deserialize_templates",
]

