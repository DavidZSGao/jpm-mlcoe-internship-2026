"""Identity-preserving balance sheet projection helpers.

This module encodes a minimal deterministic backbone inspired by the
Vélez-Pareja literature: given high-level drivers (sales growth, margin,
capital expenditure ratio, payout policy) it advances the balance sheet
while guaranteeing that Assets = Liabilities + Equity at every step.

The intent is to expose building blocks that can be composed with
TensorFlow models or classical forecasting pipelines. A neural network
can learn to predict the driver vectors, while the functions here enforce
accounting structure and return fully specified statements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class BalanceSheetState:
    """Snapshot of the balance sheet at a point in time."""

    cash: float
    receivables: float
    inventory: float
    other_current_assets: float
    net_pp_and_e: float
    other_non_current_assets: float
    accounts_payable: float
    short_term_debt: float
    accrued_expenses: float
    long_term_debt: float
    other_liabilities: float
    equity: float

    def total_assets(self) -> float:
        current_assets = self.cash + self.receivables + self.inventory + self.other_current_assets
        non_current_assets = self.net_pp_and_e + self.other_non_current_assets
        return current_assets + non_current_assets

    def total_liabilities(self) -> float:
        current_liabilities = self.accounts_payable + self.short_term_debt + self.accrued_expenses
        return current_liabilities + self.long_term_debt + self.other_liabilities

    def as_dict(self) -> Dict[str, float]:
        return {
            "cash": self.cash,
            "receivables": self.receivables,
            "inventory": self.inventory,
            "other_current_assets": self.other_current_assets,
            "net_pp_and_e": self.net_pp_and_e,
            "other_non_current_assets": self.other_non_current_assets,
            "accounts_payable": self.accounts_payable,
            "short_term_debt": self.short_term_debt,
            "accrued_expenses": self.accrued_expenses,
            "long_term_debt": self.long_term_debt,
            "other_liabilities": self.other_liabilities,
            "equity": self.equity,
        }


@dataclass
class DriverVector:
    """High-level operational and financing drivers for one period."""

    sales: float
    sales_growth: float
    ebit_margin: float
    tax_rate: float
    depreciation: float
    capex_ratio: float  # Capex as proportion of sales
    nwc_ratio: float  # Net working capital (ex cash / debt) as proportion of sales
    payout_ratio: float  # Fraction of net income paid as dividends
    target_debt_ratio: float  # Debt / (Debt + Equity)


@dataclass
class ProjectionResult:
    state: BalanceSheetState
    income_statement: Dict[str, float]
    cash_flow_statement: Dict[str, float]
    identity_gap: float


def project_forward(previous: BalanceSheetState, drivers: DriverVector) -> ProjectionResult:
    """Advance the balance sheet by one period using deterministic schedules.

    The sequence follows a simple flow:
    1. Derive income statement metrics from sales and margins.
    2. Compute target invested capital via working-capital and capex rules.
    3. Update financing to hit the target leverage while distributing dividends.
    4. Record cash movement so assets equal liabilities + equity.

    Returns a :class:`ProjectionResult` containing the new state, per-period
    income statement line items, cash flow summary, and the residual identity
    gap (should be numerically close to zero).
    """

    sales = drivers.sales * (1.0 + drivers.sales_growth)
    ebit = sales * drivers.ebit_margin

    # Working capital requirement excluding cash and short-term debt
    nwc = max(0.0, sales * drivers.nwc_ratio)

    # Maintain working capital split between receivables, inventory, payables
    receivables = nwc * 0.4
    inventory = nwc * 0.4
    accounts_payable = nwc * 0.2

    capex = sales * drivers.capex_ratio
    net_pp_and_e = max(0.0, previous.net_pp_and_e + capex - drivers.depreciation)

    other_current_assets = previous.other_current_assets  # keep static for now
    other_non_current_assets = previous.other_non_current_assets
    accrued_expenses = previous.accrued_expenses
    other_liabilities = previous.other_liabilities

    delta_receivables = receivables - previous.receivables
    delta_inventory = inventory - previous.inventory
    delta_payables = accounts_payable - previous.accounts_payable
    delta_nwc = delta_receivables + delta_inventory - delta_payables

    # Financing policy: solve for debt/equity that matches target ratio.
    invested_capital = (
        receivables + inventory + other_current_assets + net_pp_and_e + other_non_current_assets
        - accounts_payable - accrued_expenses - other_liabilities
    )
    target_debt = max(0.0, drivers.target_debt_ratio * invested_capital)
    long_term_debt = target_debt * 0.8
    short_term_debt = target_debt * 0.2

    # Interest expense and tax shield
    average_debt = (previous.long_term_debt + long_term_debt + previous.short_term_debt + short_term_debt) / 2.0
    interest_rate = 0.05
    interest_expense = average_debt * interest_rate

    nopat = ebit * (1.0 - drivers.tax_rate)
    fcf = nopat + drivers.depreciation - capex - delta_nwc

    ebt = ebit - interest_expense
    taxes = max(0.0, ebt) * drivers.tax_rate
    net_income = ebt - taxes
    dividends = max(0.0, net_income * drivers.payout_ratio)
    retained_earnings = net_income - dividends

    equity = previous.equity + retained_earnings

    previous_debt_total = previous.short_term_debt + previous.long_term_debt
    new_debt_total = short_term_debt + long_term_debt

    non_cash_assets = receivables + inventory + other_current_assets + net_pp_and_e + other_non_current_assets
    total_liabilities_equity = accounts_payable + accrued_expenses + other_liabilities + new_debt_total + equity
    cash = max(0.0, total_liabilities_equity - non_cash_assets)

    cash_delta = cash - previous.cash
    operating_cf = nopat + drivers.depreciation - delta_nwc
    investing_cf = -capex
    financing_cf = cash_delta - operating_cf - investing_cf
    expected_financing_cf = (new_debt_total - previous_debt_total) - dividends

    new_state = BalanceSheetState(
        cash=cash,
        receivables=receivables,
        inventory=inventory,
        other_current_assets=other_current_assets,
        net_pp_and_e=net_pp_and_e,
        other_non_current_assets=other_non_current_assets,
        accounts_payable=accounts_payable,
        short_term_debt=short_term_debt,
        accrued_expenses=accrued_expenses,
        long_term_debt=long_term_debt,
        other_liabilities=other_liabilities,
        equity=equity,
    )

    income_statement = {
        "sales": sales,
        "ebit": ebit,
        "interest_expense": interest_expense,
        "taxes": taxes,
        "net_income": net_income,
        "dividends": dividends,
    }

    cash_flow_statement = {
        "free_cash_flow": fcf,
        "operating_cash_flow": operating_cf,
        "investing_cash_flow": investing_cf,
        "financing_cash_flow": financing_cf,
        "financing_policy_flow": expected_financing_cf,
        "financing_gap": financing_cf - expected_financing_cf,
    }

    identity_gap = new_state.total_assets() - (new_state.total_liabilities() + new_state.equity)

    return ProjectionResult(
        state=new_state,
        income_statement=income_statement,
        cash_flow_statement=cash_flow_statement,
        identity_gap=identity_gap,
    )


def project_horizon(initial_state: BalanceSheetState, driver_sequence: Tuple[DriverVector, ...]) -> Tuple[ProjectionResult, ...]:
    """Project multiple periods sequentially, carrying forward the state."""

    results = []
    current_state = initial_state
    for drivers in driver_sequence:
        result = project_forward(current_state, drivers)
        current_state = result.state
        results.append(result)
    return tuple(results)


__all__ = [
    "BalanceSheetState",
    "DriverVector",
    "ProjectionResult",
    "project_forward",
    "project_horizon",
]
