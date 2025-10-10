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
    "net_interest_income": ("netInterestIncome",),
    "interest_income": ("interestIncome",),
    "interest_expense": ("interestExpense",),
    "tangible_equity": ("tangibleBookValue", "netTangibleAssets"),
}


BASE_FEATURES = list(DriverFeatures.__annotations__.keys())

OPTIONAL_FEATURES = [
    "tangible_equity_ratio",
    "net_interest_margin",
    "interest_income_ratio",
    "interest_expense_ratio",
    "asset_growth",
    "equity_growth",
    "net_income_growth",
    "tangible_equity_growth",
]


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

    depreciation = pd.to_numeric(
        _pick(cashflow, _CANDIDATES['depreciation']), errors='coerce'
    ).abs()
    capex = pd.to_numeric(_pick(cashflow, _CANDIDATES['capex']), errors='coerce').abs()

    receivables = pd.to_numeric(_pick(balance, _CANDIDATES['receivables']), errors='coerce')
    inventory = pd.to_numeric(_pick(balance, _CANDIDATES['inventory']), errors='coerce')
    payables = pd.to_numeric(_pick(balance, _CANDIDATES['payables']), errors='coerce')

    nwc = receivables + inventory - payables
    nwc_ratio = nwc.divide(sales)

    depreciation_ratio = depreciation.divide(sales)
    capex_ratio = capex.divide(sales)

    net_income = pd.to_numeric(_pick(income, _CANDIDATES['net_income']), errors='coerce')
    dividends = pd.to_numeric(_pick(cashflow, _CANDIDATES['dividends']), errors='coerce').abs()
    payout_ratio = dividends.divide(net_income).replace([pd.NA, pd.NaT], 0)

    debt = pd.to_numeric(_pick(balance, _CANDIDATES['debt']), errors='coerce')
    equity = pd.to_numeric(_pick(balance, _CANDIDATES['equity']), errors='coerce')
    leverage_ratio = debt.divide(debt + equity)
    sales_per_asset = sales.divide(assets)

    net_interest_income = pd.to_numeric(
        _pick(income, _CANDIDATES["net_interest_income"]), errors="coerce"
    )
    interest_income = pd.to_numeric(
        _pick(income, _CANDIDATES["interest_income"]), errors="coerce"
    )
    interest_expense = pd.to_numeric(
        _pick(income, _CANDIDATES["interest_expense"]), errors="coerce"
    ).abs()
    tangible_equity = pd.to_numeric(
        _pick(balance, _CANDIDATES["tangible_equity"]), errors="coerce"
    )

    if net_interest_income.replace(0, pd.NA).isna().all():
        net_interest_income = interest_income.subtract(interest_expense, fill_value=0)

    net_interest_margin = net_interest_income.divide(assets)
    interest_income_ratio = interest_income.divide(assets)
    interest_expense_ratio = interest_expense.divide(assets)
    tangible_equity_ratio = tangible_equity.divide(assets)

    asset_growth = assets.pct_change(fill_method=None)
    equity_growth = equity.pct_change(fill_method=None)
    net_income_growth = net_income.pct_change(fill_method=None)
    tangible_equity_growth = tangible_equity.pct_change(fill_method=None)

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
        'tangible_equity_ratio': tangible_equity_ratio,
        'net_interest_margin': net_interest_margin,
        'interest_income_ratio': interest_income_ratio,
        'interest_expense_ratio': interest_expense_ratio,
        'asset_growth': asset_growth,
        'equity_growth': equity_growth,
        'net_income_growth': net_income_growth,
        'tangible_equity_growth': tangible_equity_growth,
    })
    cleaned = out.apply(pd.to_numeric, errors='coerce')
    cleaned = cleaned.mask(~np.isfinite(cleaned))
    cleaned = cleaned.dropna(subset=BASE_FEATURES)
    fill_defaults = {feature: 0.0 for feature in OPTIONAL_FEATURES if feature in cleaned.columns}
    if fill_defaults:
        cleaned = cleaned.fillna(value=fill_defaults)
    return cleaned


def augment_with_lagged_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: int,
    *,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Append lagged versions of the specified columns for each ticker."""

    if lags <= 0 or not columns:
        return df

    augmented = df.copy()
    if 'period' in augmented.columns:
        augmented['period'] = pd.to_datetime(augmented['period'])

    augmented = augmented.sort_values(['ticker', 'period']).reset_index(drop=True)

    generated: list[str] = []
    for column in columns:
        if column not in augmented.columns:
            continue
        for lag in range(1, lags + 1):
            lagged_name = f"{column}_lag{lag}"
            augmented[lagged_name] = augmented.groupby('ticker')[column].shift(lag)
            generated.append(lagged_name)

    if drop_missing and generated:
        augmented = augmented.dropna(subset=generated)

    return augmented


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
    "BASE_FEATURES",
    "OPTIONAL_FEATURES",
    "compute_driver_frame",
    "augment_with_lagged_features",
    "compute_drivers_for_ticker",
    "compute_drivers",
]
