"""Utilities to map processed statements into BalanceSheetState objects."""

from __future__ import annotations

from typing import Dict
from pathlib import Path

import pandas as pd

from mlcoe_q1.models.balance_sheet_constraints import BalanceSheetState
from .statement_loader import load_processed_statement, wide_pivot


def _normalize_label(label: str) -> str:
    return ''.join(ch for ch in label.lower() if ch.isalnum())


def _series(df: pd.DataFrame, *candidates: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    normalized = {_normalize_label(column): column for column in df.columns}
    for name in candidates:
        key = _normalize_label(name)
        if key in normalized:
            return df[normalized[key]]
    return pd.Series(0.0, index=df.index, dtype=float)


def extract_state(df: pd.DataFrame, period: pd.Timestamp) -> BalanceSheetState:
    balance = wide_pivot(df, 'balance_sheet')
    period = pd.to_datetime(period)

    cash_series = _series(balance, 'cashAndCashEquivalents', 'cashCashEquivalentsAndShortTermInvestments')
    receivables_series = _series(balance, 'accountsReceivable', 'netReceivables')
    inventory_series = _series(balance, 'inventory')
    current_assets_series = _series(balance, 'currentAssets')
    current_liabilities_series = _series(balance, 'currentLiabilities')
    other_current_assets_series = _series(balance, 'otherCurrentAssets')
    accounts_payable_series = _series(balance, 'accountsPayable')
    current_debt_series = _series(balance, 'currentDebt', 'shortLongTermDebtTotal')
    other_current_liabilities_series = _series(balance, 'otherCurrentLiabilities')
    long_term_debt_series = _series(balance, 'longTermDebt', 'longTermDebtAndCapitalLeaseObligation')
    other_liabilities_series = _series(balance, 'otherNonCurrentLiabilities')
    total_liabilities_series = _series(balance, 'totalLiabilitiesNetMinorityInterest')
    total_assets_series = _series(balance, 'totalAssets')
    accumulated_dep_series = _series(balance, 'accumulatedDepreciation')
    gross_ppe_series = _series(balance, 'grossPPE')
    equity_series = _series(balance, 'totalStockholderEquity', 'commonStockEquity')

    cash = float(cash_series.get(period, 0.0))
    receivables = float(receivables_series.get(period, 0.0))
    inventory = float(inventory_series.get(period, 0.0))

    other_current_assets = float(
        other_current_assets_series.get(period, current_assets_series.get(period, 0.0) - cash - receivables - inventory)
    )

    net_ppe = float((gross_ppe_series - accumulated_dep_series).get(period, 0.0))
    other_non_current_assets = float(
        total_assets_series.get(period, 0.0)
        - current_assets_series.get(period, 0.0)
        - net_ppe
    )

    accounts_payable = float(accounts_payable_series.get(period, 0.0))
    short_term_debt = float(current_debt_series.get(period, 0.0))
    accrued_expenses = float(
        other_current_liabilities_series.get(period, current_liabilities_series.get(period, 0.0) - accounts_payable - short_term_debt)
    )

    long_term_debt = float(long_term_debt_series.get(period, 0.0))

    total_liabilities = float(total_liabilities_series.get(period, 0.0))
    other_liabilities = max(0.0, total_liabilities - (accounts_payable + short_term_debt + accrued_expenses + long_term_debt))

    equity = float(equity_series.get(period, total_assets_series.get(period, 0.0) - total_liabilities))

    return BalanceSheetState(
        cash=cash,
        receivables=receivables,
        inventory=inventory,
        other_current_assets=other_current_assets,
        net_pp_and_e=net_ppe,
        other_non_current_assets=other_non_current_assets,
        accounts_payable=accounts_payable,
        short_term_debt=short_term_debt,
        accrued_expenses=accrued_expenses,
        long_term_debt=long_term_debt,
        other_liabilities=other_liabilities,
        equity=equity,
    )


def extract_states(path: Path) -> Dict[pd.Timestamp, BalanceSheetState]:
    df = load_processed_statement(path)
    balance = wide_pivot(df, 'balance_sheet')
    out: Dict[pd.Timestamp, BalanceSheetState] = {}
    for period in balance.index:
        out[period] = extract_state(df, period)
    return out

__all__ = ["extract_state", "extract_states"]
