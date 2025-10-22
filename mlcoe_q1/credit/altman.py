"""Altman Z-score helpers for credit rating prototypes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class AltmanInputs:
    """Minimal set of financial metrics required for the Altman Z-score."""

    total_assets: float
    total_liabilities: float
    current_assets: float
    current_liabilities: float
    retained_earnings: float
    ebit: float
    revenue: float
    market_equity: float


@dataclass
class AltmanResult:
    """Computed Altman Z-score and intermediate ratios."""

    z_score: float
    working_capital_ratio: float
    retained_earnings_ratio: float
    ebit_ratio: float
    market_equity_ratio: float
    revenue_ratio: float


def compute_altman_z(inputs: AltmanInputs) -> AltmanResult:
    """Compute the Altman Z-score using classic 1968 coefficients.

    The implementation follows the manufacturing-focused formulation:

    ``Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5``

    where:
    - X1 = Working Capital / Total Assets
    - X2 = Retained Earnings / Total Assets
    - X3 = EBIT / Total Assets
    - X4 = Market Value of Equity / Total Liabilities
    - X5 = Revenue / Total Assets
    """

    if inputs.total_assets == 0 or inputs.total_liabilities == 0:
        raise ValueError("Total assets and liabilities must be non-zero for Altman Z-score")

    working_capital = inputs.current_assets - inputs.current_liabilities
    x1 = working_capital / inputs.total_assets
    x2 = inputs.retained_earnings / inputs.total_assets
    x3 = inputs.ebit / inputs.total_assets
    x4 = inputs.market_equity / inputs.total_liabilities
    x5 = inputs.revenue / inputs.total_assets

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    return AltmanResult(
        z_score=z,
        working_capital_ratio=x1,
        retained_earnings_ratio=x2,
        ebit_ratio=x3,
        market_equity_ratio=x4,
        revenue_ratio=x5,
    )


def assign_rating_bucket(z_score: float) -> str:
    """Map an Altman Z-score to an ordinal rating bucket."""

    if np.isnan(z_score):
        return "unavailable"
    if z_score >= 3.0:
        return "investment_grade"
    if z_score >= 2.5:
        return "bbb"
    if z_score >= 1.8:
        return "bb"
    if z_score >= 1.1:
        return "b"
    return "ccc"


def _lookup_first(series: pd.Series, candidates: Iterable[str]) -> Optional[float]:
    """Return the first available value from a set of candidate line items."""

    for name in candidates:
        if name in series.index:
            value = series[name]
            if pd.notna(value):
                return float(value)
    return None


BALANCE_SHEET_CANDIDATES: Dict[str, Iterable[str]] = {
    "total_assets": ("Total Assets",),
    "total_liabilities": (
        "Total Liabilities Net Minority Interest",
        "Total Liabilities",
    ),
    "current_assets": ("Current Assets",),
    "current_liabilities": ("Current Liabilities",),
    "retained_earnings": ("Retained Earnings",),
    "shares_outstanding": (
        "Ordinary Shares Number",
        "Share Issued",
        "Basic Average Shares",
        "Diluted Average Shares",
    ),
    "shareholders_equity": ("Stockholders Equity", "Total Equity Gross Minority Interest"),
}

INCOME_STATEMENT_CANDIDATES: Dict[str, Iterable[str]] = {
    "ebit": ("EBIT", "Operating Income", "OperatingIncome"),
    "revenue": ("Total Revenue", "TotalRevenue", "Revenue"),
}


def derive_altman_inputs(
    balance_sheet: pd.DataFrame,
    income_statement: pd.DataFrame,
    *,
    period: pd.Timestamp,
    market_equity: Optional[float] = None,
    fallback_equity: Optional[float] = None,
) -> Optional[AltmanInputs]:
    """Derive Altman inputs for a specific reporting period.

    Parameters
    ----------
    balance_sheet:
        Balance sheet dataframe with metrics as index and reporting periods as columns.
    income_statement:
        Income statement dataframe with metrics as index and reporting periods as columns.
    period:
        Timestamp referencing the column to extract.
    market_equity:
        Optional externally supplied market equity. If missing, ``fallback_equity`` is used.
    fallback_equity:
        Value to use when market equity is unavailable (e.g., book equity).
    """

    if period not in balance_sheet.columns or period not in income_statement.columns:
        return None

    bs_slice = balance_sheet[period]
    is_slice = income_statement[period]

    metrics: Dict[str, Optional[float]] = {}
    for key, candidates in BALANCE_SHEET_CANDIDATES.items():
        metrics[key] = _lookup_first(bs_slice, candidates)

    for key, candidates in INCOME_STATEMENT_CANDIDATES.items():
        metrics[key] = _lookup_first(is_slice, candidates)

    required = (
        "total_assets",
        "total_liabilities",
        "current_assets",
        "current_liabilities",
        "retained_earnings",
        "ebit",
        "revenue",
    )
    if any(metrics.get(name) in (None, 0.0) for name in required):
        return None

    equity_value: Optional[float] = market_equity
    if equity_value is None:
        equity_value = fallback_equity or metrics.get("shareholders_equity")
    if equity_value in (None, 0.0):
        return None

    return AltmanInputs(
        total_assets=metrics["total_assets"],
        total_liabilities=metrics["total_liabilities"],
        current_assets=metrics["current_assets"],
        current_liabilities=metrics["current_liabilities"],
        retained_earnings=metrics["retained_earnings"],
        ebit=metrics["ebit"],
        revenue=metrics["revenue"],
        market_equity=equity_value,
    )
