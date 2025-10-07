"""Feature engineering utilities to derive forecasting drivers from statements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd
import numpy as np

from .statement_loader import load_processed_statement, wide_pivot


def _normalize_label(label: str) -> str:
    """Lowercase a line item and strip non-alphanumeric characters for fuzzy matching."""

    return "".join(ch for ch in label.lower() if ch.isalnum())


@dataclass
class DriverFeatures:
    sales: float
    log_sales: float
    sales_per_asset: float
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
    "assets": (
        "totalAssets",
        "totalAssetsReported",
        "assets",
    ),
    "net_income": ("netIncome", "netIncomeCommonStockholders"),
}


def _pick(series_frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    if series_frame.empty:
        return pd.Series(dtype=float)

    normalized = {_normalize_label(column): column for column in series_frame.columns}
    for name in candidates:
        key = _normalize_label(name)
        if key in normalized:
            return series_frame[normalized[key]]

    return pd.Series(0.0, index=series_frame.index, dtype=float)


def compute_driver_frame(df: pd.DataFrame) -> pd.DataFrame:
    income = wide_pivot(df, 'income_statement')
    balance = wide_pivot(df, 'balance_sheet')
    cashflow = wide_pivot(df, 'cashflow_statement')

    sales = pd.to_numeric(_pick(income, _CANDIDATES['sales']), errors='coerce').replace(0, pd.NA)
    assets = pd.to_numeric(_pick(balance, _CANDIDATES['assets']), errors='coerce').replace(0, pd.NA)

    positive_sales = sales.where(sales > 0)
    log_sales = np.log(pd.to_numeric(positive_sales, errors='coerce'))

    sales_growth = sales.pct_change(fill_method=None)

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
    sales_per_asset = sales.divide(assets)

    out = pd.DataFrame({
        'sales': sales,
        'log_sales': log_sales,
        'sales_per_asset': sales_per_asset,
        'sales_growth': sales_growth,
        'ebit_margin': ebit_margin,
        'depreciation_ratio': depreciation_ratio,
        'capex_ratio': capex_ratio,
        'nwc_ratio': nwc_ratio,
        'payout_ratio': payout_ratio,
        'leverage_ratio': leverage_ratio,
    })
    cleaned = out.apply(pd.to_numeric, errors='coerce')
    cleaned = cleaned.mask(~np.isfinite(cleaned))
    return cleaned.dropna()


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
