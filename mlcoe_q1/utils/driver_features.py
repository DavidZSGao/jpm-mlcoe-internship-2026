"""Feature engineering utilities to derive forecasting drivers from statements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Sequence

import pandas as pd

from .statement_loader import load_processed_statement, wide_pivot


@dataclass
class DriverFeatures:
    sales: float
    sales_growth: float
    ebit_margin: float
    depreciation_ratio: float
    capex_ratio: float
    nwc_ratio: float
    payout_ratio: float
    leverage_ratio: float


_CANDIDATES: Dict[str, Sequence[str]] = {
    "sales": ("totalRevenue", "revenue", "operatingRevenue"),
    "ebit": ("eBIT", "ebit", "operatingIncome"),
    "depreciation": (
        "depreciationAndAmortization",
        "depreciation",
        "depreciationAmortizationDepletion",
    ),
    "capex": ("capitalExpenditure", "capitalExpenditures", "netPPEPurchaseAndSale"),
    "receivables": ("netReceivables", "accountsReceivable", "accountReceivable"),
    "inventory": ("inventory", "rawMaterials", "finishedGoods"),
    "payables": ("accountsPayable", "tradeAndOtherPayables"),
    "dividends": (
        "cashDividendsPaid",
        "commonStockDividendPaid",
        "dividendPaidCFO",
    ),
    "debt": (
        "shortLongTermDebtTotal",
        "totalDebt",
        "currentDebt",
    ),
    "equity": (
        "totalStockholderEquity",
        "stockholdersEquity",
        "commonStockEquity",
    ),
    "net_income": ("netIncome", "netIncomeCommonStockholders"),
}


def _pick(series_frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for name in candidates:
        if name in series_frame.columns:
            return series_frame[name]
    return pd.Series(0.0, index=series_frame.index)


def compute_driver_frame(df: pd.DataFrame) -> pd.DataFrame:
    income = wide_pivot(df, 'income_statement')
    balance = wide_pivot(df, 'balance_sheet')
    cashflow = wide_pivot(df, 'cashflow_statement')

    sales = _pick(income, _CANDIDATES['sales'])
    sales = sales.replace(0, pd.NA)
    sales_growth = sales.pct_change()

    ebit = _pick(income, _CANDIDATES['ebit'])
    ebit_margin = ebit.divide(sales)

    depreciation = _pick(cashflow, _CANDIDATES['depreciation']).abs()
    capex = _pick(cashflow, _CANDIDATES['capex']).abs()

    receivables = _pick(balance, _CANDIDATES['receivables'])
    inventory = _pick(balance, _CANDIDATES['inventory'])
    payables = _pick(balance, _CANDIDATES['payables'])

    nwc = receivables + inventory - payables
    nwc_ratio = nwc.divide(sales)

    depreciation_ratio = depreciation.divide(sales)
    capex_ratio = capex.divide(sales)

    net_income = _pick(income, _CANDIDATES['net_income'])
    dividends = _pick(cashflow, _CANDIDATES['dividends']).abs()
    payout_ratio = dividends.divide(net_income).replace([pd.NA, pd.NaT], 0)

    debt = _pick(balance, _CANDIDATES['debt'])
    equity = _pick(balance, _CANDIDATES['equity'])
    leverage_ratio = debt.divide(debt + equity)

    out = pd.DataFrame({
        'sales': sales,
        'sales_growth': sales_growth,
        'ebit_margin': ebit_margin,
        'depreciation_ratio': depreciation_ratio,
        'capex_ratio': capex_ratio,
        'nwc_ratio': nwc_ratio,
        'payout_ratio': payout_ratio,
        'leverage_ratio': leverage_ratio,
    })
    return out.dropna()


def compute_drivers_for_ticker(root: Path, ticker: str) -> pd.DataFrame:
    df = load_processed_statement(root / f"{ticker.upper()}.parquet")
    features = compute_driver_frame(df)
    features = features.reset_index().rename(columns={'index': 'period'})
    features['ticker'] = ticker.upper()
    return features


def compute_drivers(root: Path, tickers: Iterable[str]) -> pd.DataFrame:
    frames = [compute_drivers_for_ticker(root, t) for t in tickers]
    return pd.concat(frames, ignore_index=True)

__all__ = [
    "DriverFeatures",
    "compute_driver_frame",
    "compute_drivers_for_ticker",
    "compute_drivers",
]
