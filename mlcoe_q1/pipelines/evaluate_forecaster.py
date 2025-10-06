"""Evaluate driver forecaster with normalization, sector flags, and bank persistence fallback."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence, List

import numpy as np
import pandas as pd
import tensorflow as tf

from mlcoe_q1.models.balance_sheet_constraints import project_forward, DriverVector
from mlcoe_q1.utils.state_extractor import extract_states

FEATURE_COLUMNS = [
    'sales',
    'sales_growth',
    'ebit_margin',
    'depreciation_ratio',
    'capex_ratio',
    'nwc_ratio',
    'payout_ratio',
    'leverage_ratio',
]
AUX_FEATURES = ['is_bank']
BANK_TICKERS = {'JPM'}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drivers",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/driver_features.parquet",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/artifacts/driver_forecaster",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports/q1/artifacts/forecaster_evaluation.parquet",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _sector_vector(ticker: str) -> np.ndarray:
    return np.asarray([1.0 if ticker.upper() in BANK_TICKERS else 0.0], dtype=np.float32)


def _inverse_transform(data: dict[str, float], transform_config: dict[str, str]) -> dict[str, float]:
    result = dict(data)
    for key, method in transform_config.items():
        if method == 'log1p' and key in result:
            result[key] = np.expm1(result[key])
    return result


def row_to_driver(row: pd.Series) -> DriverVector:
    return DriverVector(
        sales=float(row['sales']),
        sales_growth=float(row['sales_growth']),
        ebit_margin=float(row['ebit_margin']),
        tax_rate=0.21,
        depreciation=float(row['sales'] * row['depreciation_ratio']),
        capex_ratio=float(row['capex_ratio']),
        nwc_ratio=float(row['nwc_ratio']),
        payout_ratio=float(np.clip(row['payout_ratio'], 0.0, 1.0)),
        target_debt_ratio=float(np.clip(row['leverage_ratio'], 0.0, 0.95)),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    df = pd.read_parquet(args.drivers)
    df = df.dropna(subset=FEATURE_COLUMNS)

    model = tf.keras.models.load_model(args.model_dir / 'model.keras')

    stats_path = args.model_dir / 'scaling.json'
    if not stats_path.exists():
        raise FileNotFoundError('Missing scaling.json from training artifacts')
    with open(stats_path) as fh:
        stats = json.load(fh)

    transform_config = stats.get('transform_config', {})
    global BANK_TICKERS
    BANK_TICKERS = set(stats.get('bank_tickers', list(BANK_TICKERS)))

    for col, method in transform_config.items():
        if col in df.columns and method == 'log1p':
            df[col] = np.log1p(df[col]).astype(float)

    x_mean = np.asarray(stats['x_mean'], dtype=np.float32)
    x_std = np.asarray(stats['x_std'], dtype=np.float32)
    y_mean = np.asarray(stats['y_mean'], dtype=np.float32)
    y_std = np.asarray(stats['y_std'], dtype=np.float32)

    records: List[dict] = []

    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('period').reset_index(drop=True)
        periods = pd.to_datetime(group['period']).to_list()
        states = extract_states(args.processed_root / f"{ticker}.parquet")
        if len(periods) < 2:
            continue

        features = group[FEATURE_COLUMNS].astype(float).to_numpy()
        sector_vec = _sector_vector(ticker)
        inputs = [np.concatenate([features[i], sector_vec]) for i in range(len(features) - 1)]
        if not inputs:
            continue
        inputs = np.asarray(inputs, dtype=np.float32)
        inputs_scaled = (inputs - x_mean) / x_std

        preds_scaled = model.predict(inputs_scaled, verbose=0)
        preds = preds_scaled * y_std + y_mean

        for i in range(len(preds)):
            prev_period = periods[i]
            target_period = periods[i + 1]
            state_prev = states.get(prev_period)
            state_true = states.get(target_period)
            if state_prev is None or state_true is None:
                continue

            if ticker.upper() in BANK_TICKERS:
                driver_data = dict(zip(FEATURE_COLUMNS, features[i + 1]))
                mode = 'persistence'
            else:
                driver_data = dict(zip(FEATURE_COLUMNS, preds[i]))
                mode = 'mlp'

            driver_data = _inverse_transform(driver_data, transform_config)
            driver_vector = row_to_driver(pd.Series(driver_data))
            result = project_forward(state_prev, driver_vector)
            pred_state = result.state

            records.append(
                {
                    'ticker': ticker,
                    'prev_period': prev_period,
                    'target_period': target_period,
                    'pred_total_assets': pred_state.total_assets(),
                    'true_total_assets': state_true.total_assets(),
                    'pred_equity': pred_state.equity,
                    'true_equity': state_true.equity,
                    'identity_gap': result.identity_gap,
                    'mode': mode,
                }
            )

    output_df = pd.DataFrame(records)
    if not output_df.empty:
        output_df['assets_mae'] = (output_df['pred_total_assets'] - output_df['true_total_assets']).abs()
        output_df['equity_mae'] = (output_df['pred_equity'] - output_df['true_equity']).abs()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    logging.info('Evaluation saved to %s', args.output)


if __name__ == '__main__':
    main()
