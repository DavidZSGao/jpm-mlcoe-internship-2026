"""Train a TensorFlow MLP to forecast driver vectors one step ahead with normalization."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf

from mlcoe_q1.models.tf_forecaster import (
    DriverConfig,
    build_mlp_forecaster,
    build_gru_forecaster,
)
from mlcoe_q1.models.bank_template import compute_bank_template, serialize_templates
from mlcoe_q1.utils.state_extractor import extract_states
from mlcoe_q1.utils.driver_features import BASE_FEATURES, OPTIONAL_FEATURES
from mlcoe_q1.pipelines import calibrate_bank_ensemble
from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config

AUX_FEATURES = ['is_bank']
BANK_TICKERS = {'JPM', 'BAC', 'C'}

TRANSFORM_CONFIG = {}


def _parse_units(raw: str) -> List[int]:
    units: List[int] = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f'Invalid GRU unit size: {token}') from exc
        if value <= 0:
            raise ValueError('GRU unit sizes must be positive integers')
        units.append(value)
    return units


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
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed",
        help="Directory containing processed statements for bank template extraction",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--architecture",
        choices=["mlp", "gru"],
        default="mlp",
        help="Neural architecture for driver forecasting",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=3,
        help="Sequence length for the GRU forecaster",
    )
    parser.add_argument(
        "--gru-units",
        type=str,
        default="64,64",
        help="Comma-separated GRU unit sizes",
    )
    parser.add_argument(
        "--recurrent-dropout",
        type=float,
        default=0.0,
        help="Recurrent dropout for GRU layers",
    )
    parser.add_argument(
        "--distribution",
        choices=["deterministic", "gaussian", "variational"],
        default="deterministic",
        help="Output distribution for the neural forecaster head",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1e-3,
        help=(
            "Weight applied to the KL divergence regulariser when --distribution "
            "is variational"
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--calibrate-banks",
        action="store_true",
        help="Fit ensemble weights blending templates and neural forecasts for banks",
    )
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


def _sector_vector(ticker: str) -> np.ndarray:
    return np.asarray([1.0 if ticker.upper() in BANK_TICKERS else 0.0], dtype=np.float32)


def _resolve_feature_columns(df: pd.DataFrame) -> list[str]:
    base = [col for col in BASE_FEATURES if col in df.columns]
    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(f'Missing required base driver features: {missing}')

    optional = [col for col in OPTIONAL_FEATURES if col in df.columns]
    extra = [
        col
        for col in df.columns
        if col not in base
        and col not in optional
        and col not in {'ticker', 'period'}
    ]
    extra.sort()
    return base + optional + extra


def build_dataset(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    architecture: str,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if architecture == 'gru':
        return _build_sequence_dataset(df, feature_columns, sequence_length)
    return _build_tabular_dataset(df, feature_columns)


def _build_tabular_dataset(
    df: pd.DataFrame, feature_columns: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_list: list[np.ndarray] = []
    aux_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('period')
        values = group[feature_columns].astype(float).to_numpy()
        sector_vec = _sector_vector(ticker)
        for i in range(len(values) - 1):
            feature_list.append(values[i])
            aux_list.append(sector_vec)
            y_list.append(values[i + 1])
    if not feature_list:
        return (
            np.empty((0, len(feature_columns)), dtype=np.float32),
            np.empty((0, len(AUX_FEATURES)), dtype=np.float32),
            np.empty((0, len(feature_columns)), dtype=np.float32),
        )
    return (
        np.asarray(feature_list, dtype=np.float32),
        np.asarray(aux_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
    )


def _build_sequence_dataset(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    aux_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    if sequence_length <= 0:
        raise ValueError('sequence_length must be positive')
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('period')
        values = group[feature_columns].astype(float).to_numpy()
        if len(values) <= sequence_length:
            continue
        sector_vec = _sector_vector(ticker)
        for idx in range(sequence_length - 1, len(values) - 1):
            start = idx - sequence_length + 1
            sequences.append(values[start:idx + 1])
            aux_list.append(sector_vec)
            y_list.append(values[idx + 1])
    if not sequences:
        return (
            np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            np.empty((0, len(AUX_FEATURES)), dtype=np.float32),
            np.empty((0, len(feature_columns)), dtype=np.float32),
        )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(aux_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
    )


def _gaussian_nll(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mean, log_var = tf.split(y_pred, 2, axis=-1)
    log_var = tf.clip_by_value(log_var, -10.0, 10.0)
    precision = tf.exp(-log_var)
    nll = 0.5 * (log_var + tf.square(y_true - mean) * precision)
    return tf.reduce_mean(nll, axis=-1)


def _gaussian_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    mean, _ = tf.split(y_pred, 2, axis=-1)
    return tf.reduce_mean(tf.abs(y_true - mean), axis=-1)


def make_variational_loss(kl_weight: float):
    kl_weight = float(kl_weight)

    def _variational_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mean, log_var = tf.split(y_pred, 2, axis=-1)
        log_var = tf.clip_by_value(log_var, -10.0, 10.0)
        precision = tf.exp(-log_var)
        nll = 0.5 * (log_var + tf.square(y_true - mean) * precision)
        nll = tf.reduce_mean(nll, axis=-1)
        kl = 0.5 * tf.reduce_sum(
            tf.square(mean) + tf.exp(log_var) - 1.0 - log_var,
            axis=-1,
        )
        return nll + kl_weight * kl

    return _variational_loss


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    df = pd.read_parquet(args.drivers)
    for column in OPTIONAL_FEATURES:
        if column not in df.columns:
            df[column] = 0.0
        else:
            df[column] = df[column].fillna(0.0)

    lag_columns = [col for col in df.columns if col.endswith(tuple(f"_lag{i}" for i in range(1, 10)))]
    if lag_columns:
        df[lag_columns] = df[lag_columns].fillna(0.0)

    feature_columns = _resolve_feature_columns(df)

    df = df.dropna(subset=feature_columns)
    architecture = args.architecture
    sequence_length = args.sequence_length if architecture == 'gru' else 1
    feature_X, aux_X, y = build_dataset(
        df,
        feature_columns,
        architecture=architecture,
        sequence_length=sequence_length,
    )

    if len(feature_X) == 0:
        raise RuntimeError('Insufficient data to train forecaster')
    cutoff = max(1, int(len(feature_X) * 0.8))
    aux_train, aux_test = aux_X[:cutoff], aux_X[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]

    if architecture == 'gru':
        feat_train = feature_X[:cutoff]
        feat_test = feature_X[cutoff:]
        flat_train = feat_train.reshape(-1, feat_train.shape[-1])
        x_mean = flat_train.mean(axis=0)
        x_std = flat_train.std(axis=0)
        x_std[x_std == 0] = 1.0
        feat_train_scaled = (feat_train - x_mean) / x_std
        feat_test_scaled = (feat_test - x_mean) / x_std if len(feat_test) else feat_test
    else:
        feat_train = feature_X[:cutoff]
        feat_test = feature_X[cutoff:]
        x_mean = feat_train.mean(axis=0)
        x_std = feat_train.std(axis=0)
        x_std[x_std == 0] = 1.0
        feat_train_scaled = (feat_train - x_mean) / x_std
        feat_test_scaled = (feat_test - x_mean) / x_std if len(feat_test) else feat_test

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std == 0] = 1.0
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    gru_units: List[int] | None = None

    if architecture == 'gru':
        train_inputs = (feat_train_scaled, aux_train)
        test_inputs = None
        if len(feat_test_scaled):
            test_inputs = (feat_test_scaled, aux_test)
        parsed_units = _parse_units(args.gru_units)
        if not parsed_units:
            raise ValueError('At least one GRU unit must be specified via --gru-units')
        gru_units = parsed_units

        model = build_gru_forecaster(
            DriverConfig(
                feature_dim=len(feature_columns),
                aux_dim=len(AUX_FEATURES),
                output_dim=len(feature_columns),
                hidden_units=[64],
                dropout=0.1,
                recurrent_dropout=args.recurrent_dropout,
                bank_feature_index=AUX_FEATURES.index('is_bank') if 'is_bank' in AUX_FEATURES else None,
                distribution=args.distribution,
            ),
            sequence_length=sequence_length,
            gru_units=gru_units,
        )
    else:
        X_train_scaled = np.concatenate([feat_train_scaled, aux_train], axis=1)
        X_test_scaled = (
            np.concatenate([feat_test_scaled, aux_test], axis=1)
            if len(feat_test_scaled)
            else feat_test_scaled
        )
        train_inputs = X_train_scaled
        test_inputs = X_test_scaled

        model = build_mlp_forecaster(
            DriverConfig(
                feature_dim=len(feature_columns),
                aux_dim=len(AUX_FEATURES),
                output_dim=len(feature_columns),
                hidden_units=[64, 64],
                dropout=0.1,
                bank_feature_index=AUX_FEATURES.index('is_bank') if 'is_bank' in AUX_FEATURES else None,
                distribution=args.distribution,
            )
        )
    if args.distribution == 'gaussian':
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=_gaussian_nll,
            metrics=[_gaussian_mae],
        )
    elif args.distribution == 'variational':
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss=make_variational_loss(args.kl_weight),
            metrics=[_gaussian_mae],
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            loss='mse',
            metrics=['mae'],
        )

    validation_data = None
    if architecture == 'gru' and test_inputs is not None and len(test_inputs[0]):
        validation_data = (test_inputs, y_test_scaled)
    elif architecture == 'mlp' and isinstance(test_inputs, np.ndarray) and len(test_inputs):
        validation_data = (test_inputs, y_test_scaled)

    history = model.fit(
        train_inputs,
        y_train_scaled,
        validation_data=validation_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / 'model.keras')

    stats = {
        'feature_columns': feature_columns,
        'aux_features': AUX_FEATURES,
        'transform_config': TRANSFORM_CONFIG,
        'bank_tickers': list(BANK_TICKERS),
        'x_mean': x_mean.tolist(),
        'x_std': x_std.tolist(),
        'aux_dim': len(AUX_FEATURES),
        'y_mean': y_mean.tolist(),
        'y_std': y_std.tolist(),
        'distribution': args.distribution,
        'architecture': architecture,
        'sequence_length': sequence_length,
    }
    if args.distribution == 'variational':
        stats['kl_weight'] = float(args.kl_weight)
    if gru_units is not None:
        stats['gru_units'] = gru_units
    with open(args.output_dir / 'scaling.json', 'w') as fh:
        json.dump(stats, fh, indent=2)

    with open(args.output_dir / 'training_history.json', 'w') as fh:
        json.dump(history.history, fh)

    bank_templates = {}
    for ticker in sorted(set(df['ticker']).intersection(BANK_TICKERS)):
        statement_path = args.processed_root / f"{ticker}.parquet"
        if not statement_path.exists():
            logging.warning("Skipping bank template for %s (missing %s)", ticker, statement_path)
            continue
        states = extract_states(statement_path)
        try:
            bank_templates[ticker] = compute_bank_template(ticker, states)
        except ValueError as exc:
            logging.warning("Unable to build bank template for %s: %s", ticker, exc)

    if bank_templates:
        with open(args.output_dir / 'bank_templates.json', 'w') as fh:
            json.dump(serialize_templates(bank_templates.values()), fh, indent=2)

    if args.calibrate_banks:
        if not bank_templates:
            logging.warning('Skipping bank ensemble calibration: no templates available')
        else:
            calibrate_bank_ensemble.main(
                [
                    '--drivers', str(args.drivers),
                    '--model-dir', str(args.output_dir),
                    '--processed-root', str(args.processed_root),
                    '--output', str(args.output_dir / 'bank_ensemble.json'),
                    '--log-level', args.log_level,
                ]
            )

    logging.info('Training complete. Model saved to %s', args.output_dir)


if __name__ == '__main__':
    main()
