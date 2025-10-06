"""Run a naive driver-based projection backtest using historical statements."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence, List

import pandas as pd
import numpy as np

from mlcoe_q1.models.balance_sheet_constraints import DriverVector, project_forward
from mlcoe_q1.utils.driver_features import compute_drivers_for_ticker
from mlcoe_q1.utils.state_extractor import extract_states


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed",
        help="Directory containing processed parquet statements",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=["GM", "JPM", "MSFT", "AAPL"],
        help="Tickers to backtest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports/q1/artifacts/baseline_backtest.parquet",
        help="Where to store evaluation results",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
    return parser.parse_args(argv)


def build_driver_vector(row: pd.Series, tax_rate: float = 0.21) -> DriverVector:
    sales = float(row['sales'])
    sales_growth = float(row.get('sales_growth', 0.0) or 0.0)
    ebit_margin = float(row.get('ebit_margin', 0.1) or 0.1)
    depreciation_ratio = float(row.get('depreciation_ratio', 0.02) or 0.02)
    capex_ratio = float(row.get('capex_ratio', 0.03) or 0.03)
    nwc_ratio = float(row.get('nwc_ratio', 0.1) or 0.1)
    payout_ratio = float(np.clip(row.get('payout_ratio', 0.3) or 0.3, 0.0, 1.0))
    leverage_ratio = float(np.clip(row.get('leverage_ratio', 0.4) or 0.4, 0.0, 0.95))

    depreciation = sales * depreciation_ratio

    return DriverVector(
        sales=sales,
        sales_growth=sales_growth,
        ebit_margin=ebit_margin,
        tax_rate=tax_rate,
        depreciation=depreciation,
        capex_ratio=capex_ratio,
        nwc_ratio=nwc_ratio,
        payout_ratio=payout_ratio,
        target_debt_ratio=leverage_ratio,
    )


def evaluate_ticker(processed_root: Path, ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()
    states = extract_states(processed_root / f"{ticker}.parquet")
    drivers = compute_drivers_for_ticker(processed_root, ticker)
    drivers = drivers.sort_values('period')

    rows: List[dict] = []
    periods = drivers['period'].tolist()
    for i in range(1, len(periods)):
        prev_period = periods[i - 1]
        target_period = periods[i]

        state_prev = states.get(prev_period)
        state_true = states.get(target_period)
        if state_prev is None or state_true is None:
            continue

        driver_row = drivers.iloc[i - 1]
        driver_vector = build_driver_vector(driver_row)
        result = project_forward(state_prev, driver_vector)
        pred_state = result.state

        rows.append(
            {
                'ticker': ticker,
                'prev_period': prev_period,
                'target_period': target_period,
                'pred_total_assets': pred_state.total_assets(),
                'true_total_assets': state_true.total_assets(),
                'pred_equity': pred_state.equity,
                'true_equity': state_true.equity,
                'pred_cash': pred_state.cash,
                'true_cash': state_true.cash,
                'identity_gap': result.identity_gap,
                'financing_gap': result.cash_flow_statement.get('financing_gap', 0.0),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    frames = [evaluate_ticker(args.processed_root, t) for t in args.tickers]
    results = pd.concat(frames, ignore_index=True)

    if not results.empty:
        results['assets_mae'] = (results['pred_total_assets'] - results['true_total_assets']).abs()
        results['equity_mae'] = (results['pred_equity'] - results['true_equity']).abs()
        results['cash_mae'] = (results['pred_cash'] - results['true_cash']).abs()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.output, index=False)
    logging.info("Backtest results saved to %s", args.output)


if __name__ == "__main__":
    main()
