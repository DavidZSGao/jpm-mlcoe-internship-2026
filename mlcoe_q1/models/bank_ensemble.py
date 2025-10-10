"""Bank ensemble utilities to blend template and neural forecasts."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from .balance_sheet_constraints import BalanceSheetState, ProjectionResult
from .bank_template import ASSET_FIELDS, LIABILITY_FIELDS


@dataclass(frozen=True)
class BankEnsembleWeights:
    """Linear combination weights for blending bank projections."""

    ticker: str
    assets_template_weight: float
    assets_mlp_weight: float
    assets_bias: float
    equity_template_weight: float
    equity_mlp_weight: float
    equity_bias: float

    def combine(
        self,
        template: ProjectionResult,
        mlp: ProjectionResult,
    ) -> ProjectionResult:
        """Blend template and neural projections into a calibrated state."""

        template_state = template.state
        mlp_state = mlp.state

        template_assets = template_state.total_assets()
        mlp_assets = mlp_state.total_assets()
        assets_total = (
            self.assets_template_weight * template_assets
            + self.assets_mlp_weight * mlp_assets
            + self.assets_bias
        )

        template_equity = template_state.equity
        mlp_equity = mlp_state.equity
        equity_total = (
            self.equity_template_weight * template_equity
            + self.equity_mlp_weight * mlp_equity
            + self.equity_bias
        )

        assets_total = float(np.clip(assets_total, 0.0, np.finfo(np.float64).max))
        equity_total = float(np.clip(equity_total, 0.0, assets_total))
        liabilities_total = max(0.0, assets_total - equity_total)

        assets = _scale_fields(template_state, mlp_state, ASSET_FIELDS, assets_total)
        liabilities = _scale_fields(
            template_state,
            mlp_state,
            LIABILITY_FIELDS,
            liabilities_total,
        )

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
            equity=equity_total,
        )

        identity_gap = state.total_assets() - (state.total_liabilities() + state.equity)
        return ProjectionResult(
            state=state,
            income_statement=mlp.income_statement,
            cash_flow_statement=mlp.cash_flow_statement,
            identity_gap=identity_gap,
        )


def _scale_fields(
    preferred_state: BalanceSheetState,
    fallback_state: BalanceSheetState,
    fields: Sequence[str],
    total: float,
) -> Dict[str, float]:
    distribution = _field_distribution(preferred_state, fields)
    if distribution is None:
        distribution = _field_distribution(fallback_state, fields)
    if distribution is None:
        uniform = 1.0 / len(fields) if fields else 0.0
        distribution = {field: uniform for field in fields}

    return {field: float(total * max(distribution.get(field, 0.0), 0.0)) for field in fields}


def _field_distribution(
    state: BalanceSheetState,
    fields: Sequence[str],
) -> Dict[str, float] | None:
    values = np.asarray([getattr(state, field) for field in fields], dtype=float)
    total = values.sum()
    if total <= 0.0:
        return None
    return {field: float(value / total) for field, value in zip(fields, values)}


def fit_ensemble_weights(records: Iterable[Mapping[str, float]], ticker: str) -> BankEnsembleWeights:
    """Solve for linear blending weights from historical projections."""

    template_assets: list[float] = []
    mlp_assets: list[float] = []
    true_assets: list[float] = []
    template_equity: list[float] = []
    mlp_equity: list[float] = []
    true_equity: list[float] = []

    for record in records:
        template_assets.append(float(record["template_assets"]))
        mlp_assets.append(float(record["mlp_assets"]))
        true_assets.append(float(record["true_assets"]))
        template_equity.append(float(record["template_equity"]))
        mlp_equity.append(float(record["mlp_equity"]))
        true_equity.append(float(record["true_equity"]))

    if len(template_assets) < 2:
        raise ValueError(f"Not enough records to fit ensemble weights for {ticker}")

    assets_coeffs = _solve_linear_combination(template_assets, mlp_assets, true_assets)
    equity_coeffs = _solve_linear_combination(template_equity, mlp_equity, true_equity)

    return BankEnsembleWeights(
        ticker=ticker,
        assets_template_weight=assets_coeffs[0],
        assets_mlp_weight=assets_coeffs[1],
        assets_bias=assets_coeffs[2],
        equity_template_weight=equity_coeffs[0],
        equity_mlp_weight=equity_coeffs[1],
        equity_bias=equity_coeffs[2],
    )


def _solve_linear_combination(
    template_values: Sequence[float],
    mlp_values: Sequence[float],
    targets: Sequence[float],
) -> np.ndarray:
    design = np.stack(
        [
            np.asarray(template_values, dtype=float),
            np.asarray(mlp_values, dtype=float),
            np.ones(len(template_values), dtype=float),
        ],
        axis=1,
    )
    target = np.asarray(targets, dtype=float)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coeffs


def serialize_ensemble(weights: Iterable[BankEnsembleWeights]) -> Dict[str, dict]:
    return {entry.ticker: asdict(entry) for entry in weights}


def deserialize_ensemble(payload: Mapping[str, Mapping[str, float]]) -> Dict[str, BankEnsembleWeights]:
    weights: Dict[str, BankEnsembleWeights] = {}
    for ticker, data in payload.items():
        weights[ticker.upper()] = BankEnsembleWeights(
            ticker=str(data.get("ticker", ticker)),
            assets_template_weight=float(data.get("assets_template_weight", 1.0)),
            assets_mlp_weight=float(data.get("assets_mlp_weight", 0.0)),
            assets_bias=float(data.get("assets_bias", 0.0)),
            equity_template_weight=float(data.get("equity_template_weight", 0.0)),
            equity_mlp_weight=float(data.get("equity_mlp_weight", 1.0)),
            equity_bias=float(data.get("equity_bias", 0.0)),
        )
    return weights


__all__ = [
    "BankEnsembleWeights",
    "fit_ensemble_weights",
    "serialize_ensemble",
    "deserialize_ensemble",
]

