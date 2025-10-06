"""Compute financial ratios for a given ticker using processed statements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlcoe_q1.utils.statement_loader import load_processed_statement, wide_pivot


def compute_ratios(df: pd.DataFrame) -> pd.Series:
    balance = wide_pivot(df, 'balance_sheet')
    income = wide_pivot(df, 'income_statement')
    cashflow = wide_pivot(df, 'cashflow_statement')

    latest_period = balance.index.max()

    def pick(frame: pd.DataFrame, *candidates: str) -> float:
        for name in candidates:
            if name in frame.columns:
                return float(frame.loc[latest_period, name])
        return 0.0

    total_revenue = pick(income, 'totalRevenue', 'revenue', 'operatingRevenue')
    total_expenses = pick(income, 'totalExpenses')
    net_income = pick(income, 'netIncome', 'netIncomeCommonStockholders')
    ebit = pick(income, 'eBIT', 'operatingIncome')
    ebitda = pick(income, 'eBITDA', 'normalizedEBITDA')
    interest_expense = pick(income, 'interestExpense', 'netInterestIncome')

    current_assets = pick(balance, 'currentAssets')
    inventory = pick(balance, 'inventory')
    current_liabilities = pick(balance, 'currentLiabilities')

    total_assets = pick(balance, 'totalAssets')
    total_liabilities = pick(balance, 'totalLiabilitiesNetMinorityInterest')
    equity = pick(balance, 'totalStockholderEquity', 'commonStockEquity')

    short_debt = pick(balance, 'currentDebt')
    long_debt = pick(balance, 'longTermDebt')
    total_debt = short_debt + long_debt

    quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities else float('nan')
    cost_to_income = total_expenses / total_revenue if total_revenue else float('nan')
    debt_to_equity = total_debt / equity if equity else float('nan')
    debt_to_assets = total_debt / total_assets if total_assets else float('nan')
    debt_to_capital = total_debt / (total_debt + equity) if (total_debt + equity) else float('nan')
    debt_to_ebitda = total_debt / ebitda if ebitda else float('nan')
    interest_coverage = ebit / interest_expense if interest_expense else float('nan')

    return pd.Series(
        {
            'period': latest_period,
            'total_revenue': total_revenue,
            'total_expenses': total_expenses,
            'net_income': net_income,
            'cost_to_income_ratio': cost_to_income,
            'quick_ratio': quick_ratio,
            'debt_to_equity': debt_to_equity,
            'debt_to_assets': debt_to_assets,
            'debt_to_capital': debt_to_capital,
            'debt_to_ebitda': debt_to_ebitda,
            'interest_coverage': interest_coverage,
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ticker', help='Ticker symbol (e.g., GM)')
    parser.add_argument(
        '--processed-root',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'data/processed',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'reports/q1/artifacts/ratio_summary.json',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    df = load_processed_statement(args.processed_root / f"{args.ticker.upper()}.parquet")
    ratios = compute_ratios(df)
    ratios_dict = ratios.to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as fh:
        json.dump(ratios_dict, fh, default=str, indent=2)


if __name__ == '__main__':
    main()
