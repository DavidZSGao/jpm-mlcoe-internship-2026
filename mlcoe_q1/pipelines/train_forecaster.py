"""Train a TensorFlow MLP to forecast driver vectors one step ahead with normalization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from mlcoe_q1.models.tf_forecaster import DriverConfig, build_mlp_forecaster

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

TRANSFORM_CONFIG = {
    'sales': 'log1p'
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drivers",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/driver_features.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/artifacts/driver_forecaster",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _sector_vector(ticker: str) -> np.ndarray:
    return np.asarray([1.0 if ticker.upper() in BANK_TICKERS else 0.0], dtype=np.float32)


def build_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for ticker, group in df.groupby('ticker'):
        if ticker.upper() in BANK_TICKERS:
            continue
        group = group.sort_values('period')
        values = group[FEATURE_COLUMNS].astype(float).to_numpy()
        sector_vec = _sector_vector(ticker)
        for i in range(len(values) - 1):
            X_list.append(np.concatenate([values[i], sector_vec]))
            y_list.append(values[i + 1])
    return np.asarray(X_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    df = pd.read_parquet(args.drivers)
    df = df.dropna(subset=FEATURE_COLUMNS)
    for col, method in TRANSFORM_CONFIG.items():
        if col in df.columns and method == 'log1p':
            df[col] = np.log1p(df[col]).astype(float)
    X, y = build_dataset(df)

    if len(X) == 0:
        raise RuntimeError('Insufficient data to train forecaster')

    cutoff = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:cutoff], X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0

    X_train_scaled = (X_train - x_mean) / x_std
    X_test_scaled = (X_test - x_mean) / x_std
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    model = build_mlp_forecaster(
        DriverConfig(input_dim=X.shape[1], hidden_units=[64, 64], dropout=0.1)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss='mse',
        metrics=['mae'],
    )

    history = model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled) if len(X_test_scaled) else None,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / 'model.keras')

    stats = {
        'feature_columns': FEATURE_COLUMNS,
        'aux_features': AUX_FEATURES,
        'transform_config': TRANSFORM_CONFIG,
        'bank_tickers': list(BANK_TICKERS),
        'x_mean': x_mean.tolist(),
        'x_std': x_std.tolist(),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist(),
    }
    with open(args.output_dir / 'scaling.json', 'w') as fh:
        json.dump(stats, fh, indent=2)

    with open(args.output_dir / 'training_history.json', 'w') as fh:
        json.dump(history.history, fh)

    logging.info('Training complete. Model saved to %s', args.output_dir)


if __name__ == '__main__':
    main()
