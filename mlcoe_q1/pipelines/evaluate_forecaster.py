"""Evaluate driver forecaster with normalization, sector flags, and bank persistence fallback."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence, List, Mapping

import numpy as np
import pandas as pd
import tensorflow as tf

from mlcoe_q1.models.balance_sheet_constraints import project_forward, DriverVector, ProjectionResult
from mlcoe_q1.models import tf_forecaster as _tf_forecaster  # noqa: F401 ensures custom layers are registered
from mlcoe_q1.models.bank_template import deserialize_templates, BankTemplate
from mlcoe_q1.models.bank_ensemble import deserialize_ensemble, BankEnsembleWeights
from mlcoe_q1.utils.state_extractor import extract_states, extract_income_metric_map

BANK_TICKERS = {'JPM', 'BAC', 'C'}


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
    parser.add_argument(
        "--bank-mode",
        choices=["auto", "template", "mlp", "persistence", "ensemble"],
        default="auto",
        help="Forecasting strategy for bank tickers",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _sector_vector(ticker: str, aux_features: Sequence[str]) -> np.ndarray:
    values: list[float] = []
    upper = ticker.upper()
    for feature in aux_features:
        if feature == 'is_bank':
            values.append(1.0 if upper in BANK_TICKERS else 0.0)
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _inverse_transform(data: dict[str, float], transform_config: Mapping[str, str]) -> dict[str, float]:
    result = dict(data)
    for key, method in transform_config.items():
        if method == 'log1p' and key in result:
            result[key] = np.expm1(result[key])
    return result


def _features_to_driver_series(
    feature_map: Mapping[str, float], transform_config: Mapping[str, str]
) -> pd.Series:
    data = _inverse_transform(feature_map, transform_config)
    if 'sales' not in data:
        if 'log_sales' in data and np.isfinite(data['log_sales']):
            data = dict(data)
            data['sales'] = float(np.exp(data['log_sales']))
        elif 'sales_log1p' in data and np.isfinite(data['sales_log1p']):
            data = dict(data)
            data['sales'] = float(np.expm1(data['sales_log1p']))

    keys = [
        'sales',
        'sales_growth',
        'ebit_margin',
        'depreciation_ratio',
        'capex_ratio',
        'nwc_ratio',
        'payout_ratio',
        'leverage_ratio',
    ]
    return pd.Series({key: data.get(key, np.nan) for key in keys})


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

    lag_columns = [col for col in df.columns if col.endswith(tuple(f"_lag{i}" for i in range(1, 10)))]
    if lag_columns:
        df[lag_columns] = df[lag_columns].fillna(0.0)

    model = tf.keras.models.load_model(args.model_dir / 'model.keras')

    stats_path = args.model_dir / 'scaling.json'
    if not stats_path.exists():
        raise FileNotFoundError('Missing scaling.json from training artifacts')
    with open(stats_path) as fh:
        stats = json.load(fh)

    feature_columns: list[str] = stats.get('feature_columns', [])
    aux_features: list[str] = stats.get('aux_features', [])
    transform_config = stats.get('transform_config', {})
    global BANK_TICKERS
    BANK_TICKERS = set(stats.get('bank_tickers', list(BANK_TICKERS)))

    bank_templates: dict[str, BankTemplate] = {}
    bank_ensemble: dict[str, BankEnsembleWeights] = {}
    templates_path = args.model_dir / 'bank_templates.json'
    if templates_path.exists():
        with open(templates_path) as fh:
            bank_templates = deserialize_templates(json.load(fh))

    ensemble_path = args.model_dir / 'bank_ensemble.json'
    if ensemble_path.exists():
        with open(ensemble_path) as fh:
            bank_ensemble = deserialize_ensemble(json.load(fh))

    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f'Missing required driver columns: {missing_columns}')

    df = df.dropna(subset=feature_columns)

    for col, method in transform_config.items():
        if col in df.columns and method == 'log1p':
            df[col] = np.log1p(df[col]).astype(float)

    x_mean = np.asarray(stats['x_mean'], dtype=np.float32)
    x_std = np.asarray(stats['x_std'], dtype=np.float32)
    y_mean = np.asarray(stats['y_mean'], dtype=np.float32)
    y_std = np.asarray(stats['y_std'], dtype=np.float32)
    aux_dim = int(stats.get('aux_dim', len(aux_features)))

    records: List[dict] = []

    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('period').reset_index(drop=True)
        periods = pd.to_datetime(group['period']).to_list()
        processed_path = args.processed_root / f"{ticker}.parquet"
        states = extract_states(processed_path)
        income_metrics = extract_income_metric_map(processed_path)
        template = bank_templates.get(ticker.upper())
        if len(periods) < 2:
            continue

        features = group[feature_columns].astype(float).to_numpy()
        sector_vec = _sector_vector(ticker, aux_features)
        if aux_dim and len(sector_vec) != aux_dim:
            raise ValueError(
                f"Auxiliary feature dimension mismatch: expected {aux_dim}, got {len(sector_vec)}"
            )

        inputs = []
        for i in range(len(features) - 1):
            feature_scaled = (features[i] - x_mean) / x_std
            inputs.append(np.concatenate([feature_scaled, sector_vec]))
        if not inputs:
            continue
        inputs = np.asarray(inputs, dtype=np.float32)
        preds_scaled = model.predict(inputs, verbose=0)
        preds = preds_scaled * y_std + y_mean

        for i in range(len(preds)):
            prev_period = periods[i]
            target_period = periods[i + 1]
            state_prev = states.get(prev_period)
            state_true = states.get(target_period)
            metrics_true = income_metrics.get(target_period)
            if state_prev is None or state_true is None or metrics_true is None:
                continue

            pred_map = {
                feature_columns[j]: float(preds[i, j])
                for j in range(len(feature_columns))
            }
            pred_series = _features_to_driver_series(pred_map, transform_config)

            upper = ticker.upper()
            template_available = template is not None
            bank_mode = args.bank_mode

            template_result: ProjectionResult | None = None
            if template_available:
                template_result = template.project(state_prev)

            mlp_result: ProjectionResult | None = None
            driver_series = pred_series
            if driver_series.isna().any():
                logging.debug('Skipping %s %s due to missing predicted drivers', ticker, target_period)
                continue
            driver_data = driver_series.to_dict()
            driver_vector = row_to_driver(pd.Series(driver_data))
            mlp_result = project_forward(state_prev, driver_vector)

            if upper in BANK_TICKERS:
                if bank_mode in {'ensemble', 'auto'} and upper in bank_ensemble and template_result is not None and mlp_result is not None:
                    weights = bank_ensemble[upper]
                    result = weights.combine(template_result, mlp_result)
                    pred_state = result.state
                    identity_gap = result.identity_gap
                    mode = 'bank_ensemble'
                elif (
                    bank_mode == 'template'
                    or (bank_mode == 'auto' and template_result is not None)
                ) and template_result is not None:
                    result = template_result
                    pred_state = result.state
                    identity_gap = result.identity_gap
                    mode = 'bank_template'
                elif bank_mode in {'persistence', 'auto'} and bank_mode != 'mlp':
                    observed_map = {
                        feature_columns[j]: float(features[i + 1, j])
                        for j in range(len(feature_columns))
                    }
                    driver_series_obs = _features_to_driver_series(observed_map, transform_config)
                    if driver_series_obs.isna().any():
                        logging.debug('Skipping %s %s due to missing observed drivers', ticker, target_period)
                        continue
                    driver_vector_obs = row_to_driver(driver_series_obs)
                    result = project_forward(state_prev, driver_vector_obs)
                    pred_state = result.state
                    identity_gap = result.identity_gap
                    mode = 'persistence'
                else:
                    result = mlp_result
                    pred_state = result.state
                    identity_gap = result.identity_gap
                    mode = 'mlp'
            else:
                result = mlp_result
                pred_state = result.state
                identity_gap = result.identity_gap
                mode = 'mlp'

            pred_net_income = float('nan')
            if result.income_statement:
                net_income_val = result.income_statement.get('net_income')
                if net_income_val is not None:
                    pred_net_income = float(net_income_val)

            records.append(
                {
                    'ticker': ticker,
                    'prev_period': prev_period,
                    'target_period': target_period,
                    'pred_total_assets': pred_state.total_assets(),
                    'true_total_assets': state_true.total_assets(),
                    'pred_equity': pred_state.equity,
                    'true_equity': state_true.equity,
                    'identity_gap': identity_gap,
                    'mode': mode,
                    'pred_net_income': pred_net_income,
                    'true_net_income': metrics_true.net_income,
                }
            )

    output_df = pd.DataFrame(records)
    if not output_df.empty:
        output_df['assets_mae'] = (output_df['pred_total_assets'] - output_df['true_total_assets']).abs()
        output_df['equity_mae'] = (output_df['pred_equity'] - output_df['true_equity']).abs()
        valid_income = output_df[['pred_net_income', 'true_net_income']].notna().all(axis=1)
        output_df.loc[valid_income, 'net_income_mae'] = (
            output_df.loc[valid_income, 'pred_net_income']
            - output_df.loc[valid_income, 'true_net_income']
        ).abs()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    logging.info('Evaluation saved to %s', args.output)


if __name__ == '__main__':
    main()
