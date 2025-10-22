"""Evaluate driver forecaster with normalization, sector flags, and bank persistence fallback."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence, List, Mapping, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from mlcoe_q1.models.balance_sheet_constraints import project_forward, DriverVector, ProjectionResult
from mlcoe_q1.models import tf_forecaster as _tf_forecaster  # noqa: F401 ensures custom layers are registered
from mlcoe_q1.models.bank_template import deserialize_templates, BankTemplate
from mlcoe_q1.models.bank_ensemble import deserialize_ensemble, BankEnsembleWeights
from mlcoe_q1.utils.state_extractor import extract_states, extract_income_metric_map
from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config

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
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=0,
        help=(
            "Number of Monte Carlo dropout samples to draw for predictive "
            "intervals (0 disables sampling)"
        ),
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.1,0.5,0.9",
        help=(
            "Comma-separated quantiles (between 0 and 1) to report when "
            "--mc-samples is enabled"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Number of sequential steps to evaluate via recursive rollouts",
    )
    parser.add_argument("--log-level", default="INFO")
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


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


def _parse_quantiles(raw: object) -> list[float]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = text
            else:
                return _parse_quantiles(parsed)
        tokens = [token.strip() for token in text.split(",")]
        values: list[float] = []
        for token in tokens:
            if not token:
                continue
            try:
                quantile = float(token)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid quantile value: {token}") from exc
            if not 0.0 <= quantile <= 1.0:
                raise ValueError(
                    f"Quantiles must fall within [0, 1]; received {quantile}"
                )
            values.append(quantile)
        return sorted(set(values))

    if isinstance(raw, Iterable):
        values: list[float] = []
        for item in raw:
            values.extend(_parse_quantiles(item))
        return sorted(set(values))

    try:
        quantile = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid quantile value: {raw!r}") from exc
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Quantiles must fall within [0, 1]; received {quantile}")
    return [quantile]


VARIANCE_DISTRIBUTIONS = {"gaussian", "variational"}


def _decode_predictions(
    preds_scaled: np.ndarray,
    distribution: str,
    output_dim: int,
) -> Tuple[np.ndarray, np.ndarray | None]:
    distribution = distribution.lower()
    if distribution in VARIANCE_DISTRIBUTIONS:
        if preds_scaled.shape[-1] != output_dim * 2:
            raise ValueError(
                "Expected probabilistic head to emit 2 * output_dim values per timestep"
            )
        mean = preds_scaled[:, :output_dim]
        log_var = preds_scaled[:, output_dim:]
        return mean, log_var
    if preds_scaled.shape[-1] != output_dim:
        raise ValueError(
            "Deterministic head output dimension mismatch: "
            f"expected {output_dim}, received {preds_scaled.shape[-1]}"
        )
    return preds_scaled, None


def _project_from_features(
    feature_values: Iterable[float],
    feature_columns: Sequence[str],
    transform_config: Mapping[str, str],
    state_prev,
) -> ProjectionResult | None:
    feature_iter = list(feature_values)
    if len(feature_iter) != len(feature_columns):
        raise ValueError(
            "Feature value length mismatch: "
            f"expected {len(feature_columns)}, received {len(feature_iter)}"
        )
    pred_map = {
        feature_columns[j]: float(feature_iter[j])
        for j in range(len(feature_columns))
    }
    pred_series = _features_to_driver_series(pred_map, transform_config)
    if pred_series.isna().any():
        return None
    driver_vector = row_to_driver(pd.Series(pred_series))
    return project_forward(state_prev, driver_vector)


def _extract_net_income(
    income_statement: Mapping[str, float] | None,
) -> float | None:
    if not income_statement:
        return None
    value = income_statement.get('net_income')
    if value is None:
        return None
    return float(value)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.mc_samples < 0:
        raise ValueError("--mc-samples must be non-negative")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    quantiles = _parse_quantiles(args.quantiles)

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
    distribution = str(stats.get('distribution', 'deterministic')).lower()
    architecture = str(stats.get('architecture', 'mlp')).lower()
    sequence_length = int(stats.get('sequence_length', 1))
    if sequence_length <= 0:
        sequence_length = 1
    if architecture not in {'mlp', 'gru'}:
        raise ValueError(f"Unsupported architecture in scaling stats: {architecture}")
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
        if len(periods) <= sequence_length:
            continue

        features = group[feature_columns].astype(float).to_numpy()
        sector_vec = _sector_vector(ticker, aux_features).astype(np.float32)
        if aux_dim and len(sector_vec) != aux_dim:
            raise ValueError(
                f"Auxiliary feature dimension mismatch: expected {aux_dim}, got {len(sector_vec)}"
            )

        features_scaled = ((features - x_mean) / x_std).astype(np.float32)

        for start_idx in range(sequence_length - 1, len(features) - 1):
            max_horizon = min(args.horizon, len(features) - start_idx - 1)
            if max_horizon <= 0:
                continue

            base_state = states.get(periods[start_idx])
            if base_state is None:
                continue

            seq_window = [
                np.asarray(features_scaled[j], dtype=np.float32)
                for j in range(start_idx - sequence_length + 1, start_idx + 1)
            ]
            current_state = base_state

            for step in range(1, max_horizon + 1):
                target_idx = start_idx + step
                target_period = periods[target_idx]
                state_true = states.get(target_period)
                metrics_true = income_metrics.get(target_period)
                if state_true is None or metrics_true is None:
                    continue

                if architecture == 'gru':
                    seq_input = np.stack(seq_window, axis=0)[None, :, :]
                    aux_input = sector_vec[None, :]
                    preds_scaled_raw = model.predict([seq_input, aux_input], verbose=0)[0]
                else:
                    feature_input = seq_window[-1]
                    model_input = np.concatenate([feature_input, sector_vec])
                    preds_scaled_raw = model.predict(model_input[None, :], verbose=0)[0]

                preds_mean_scaled, log_var_scaled = _decode_predictions(
                    preds_scaled_raw[None, :], distribution, len(feature_columns)
                )
                preds_mean_scaled = preds_mean_scaled[0]
                preds = preds_mean_scaled * y_std + y_mean

                mc_preds: np.ndarray | None = None
                if args.mc_samples:
                    if distribution in VARIANCE_DISTRIBUTIONS and log_var_scaled is not None:
                        std_scaled = np.exp(0.5 * np.clip(log_var_scaled[0], -10.0, 10.0))
                        epsilon = np.random.standard_normal(
                            size=(args.mc_samples, len(preds_mean_scaled))
                        )
                        sample_scaled = preds_mean_scaled[None, :] + epsilon * std_scaled[None, :]
                        mc_preds = sample_scaled * y_std + y_mean
                    else:
                        sample_values = []
                        for _ in range(args.mc_samples):
                            if architecture == 'gru':
                                sample_raw = model(
                                    [seq_input, aux_input], training=True
                                ).numpy()[0]
                            else:
                                sample_raw = model(
                                    model_input[None, :], training=True
                                ).numpy()[0]
                            sample_values.append(sample_raw)
                        if sample_values:
                            sample_scaled = np.stack(sample_values, axis=0)
                            mc_preds = sample_scaled * y_std + y_mean

                state_prev = current_state
                template_result: ProjectionResult | None = None
                if template is not None:
                    template_result = template.project(state_prev)

                mlp_result = _project_from_features(
                    preds, feature_columns, transform_config, state_prev
                )
                if mlp_result is None:
                    logging.debug(
                        'Skipping %s %s due to missing predicted drivers',
                        ticker,
                        target_period,
                    )
                    continue

                persistence_result: ProjectionResult | None = None
                upper = ticker.upper()
                bank_mode = args.bank_mode
                if (
                    upper in BANK_TICKERS
                    and bank_mode in {'persistence', 'auto'}
                    and bank_mode != 'mlp'
                ):
                    observed_map = {
                        feature_columns[j]: float(features[target_idx, j])
                        for j in range(len(feature_columns))
                    }
                    driver_series_obs = _features_to_driver_series(
                        observed_map, transform_config
                    )
                    if driver_series_obs.isna().any():
                        logging.debug(
                            'Skipping %s %s persistence fallback due to missing observed drivers',
                            ticker,
                            target_period,
                        )
                    else:
                        driver_vector_obs = row_to_driver(driver_series_obs)
                        persistence_result = project_forward(state_prev, driver_vector_obs)

                ensemble_weights = bank_ensemble.get(upper)
                if upper in BANK_TICKERS:
                    if (
                        bank_mode in {'ensemble', 'auto'}
                        and ensemble_weights is not None
                        and template_result is not None
                    ):
                        result = ensemble_weights.combine(template_result, mlp_result)
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
                    elif (
                        bank_mode in {'persistence', 'auto'}
                        and bank_mode != 'mlp'
                        and persistence_result is not None
                    ):
                        result = persistence_result
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

                net_income_val = _extract_net_income(result.income_statement)
                pred_net_income = (
                    float(net_income_val) if net_income_val is not None else float('nan')
                )

                record: dict[str, object] = {
                    'ticker': ticker,
                    'prev_period': periods[start_idx + step - 1],
                    'target_period': target_period,
                    'horizon': step,
                    'pred_total_assets': pred_state.total_assets(),
                    'true_total_assets': state_true.total_assets(),
                    'pred_equity': pred_state.equity,
                    'true_equity': state_true.equity,
                    'identity_gap': identity_gap,
                    'mode': mode,
                    'pred_net_income': pred_net_income,
                    'true_net_income': metrics_true.net_income,
                    'distribution': distribution,
                }

                if mc_preds is not None:
                    sample_assets: list[float] = []
                    sample_equity: list[float] = []
                    sample_identity: list[float] = []
                    sample_net_income: list[float] = []

                    if mode in {'bank_template', 'persistence'}:
                        sample_assets = [pred_state.total_assets()]
                        sample_equity = [pred_state.equity]
                        sample_identity = [identity_gap]
                        if np.isfinite(pred_net_income):
                            sample_net_income = [float(pred_net_income)]
                    else:
                        for sample_idx in range(mc_preds.shape[0]):
                            sample_result = _project_from_features(
                                mc_preds[sample_idx],
                                feature_columns,
                                transform_config,
                                state_prev,
                            )
                            if sample_result is None:
                                continue
                            sample_projection = sample_result
                            if (
                                mode == 'bank_ensemble'
                                and template_result is not None
                                and ensemble_weights is not None
                            ):
                                sample_projection = ensemble_weights.combine(
                                    template_result, sample_result
                                )

                            sample_state = sample_projection.state
                            sample_assets.append(sample_state.total_assets())
                            sample_equity.append(sample_state.equity)
                            sample_identity.append(sample_projection.identity_gap)
                            sample_income_val = _extract_net_income(
                                sample_projection.income_statement
                            )
                            if sample_income_val is not None:
                                sample_net_income.append(float(sample_income_val))

                    sample_count = len(sample_assets)
                    record['mc_sample_count'] = int(sample_count)
                    if distribution in VARIANCE_DISTRIBUTIONS and log_var_scaled is not None:
                        strategy = 'gaussian_head' if distribution == 'gaussian' else 'variational_head'
                    else:
                        strategy = 'dropout'
                    record['mc_strategy'] = strategy

                    if sample_assets:
                        assets_array = np.asarray(sample_assets, dtype=float)
                        record['pred_total_assets_mc_mean'] = float(np.mean(assets_array))
                        record['pred_total_assets_mc_std'] = float(np.std(assets_array, ddof=0))
                        for quantile in quantiles:
                            suffix = f"{int(round(quantile * 100)):02d}"
                            record[f"pred_total_assets_q{suffix}"] = float(
                                np.quantile(assets_array, quantile)
                            )
                    if sample_equity:
                        equity_array = np.asarray(sample_equity, dtype=float)
                        record['pred_equity_mc_mean'] = float(np.mean(equity_array))
                        record['pred_equity_mc_std'] = float(np.std(equity_array, ddof=0))
                        for quantile in quantiles:
                            suffix = f"{int(round(quantile * 100)):02d}"
                            record[f"pred_equity_q{suffix}"] = float(
                                np.quantile(equity_array, quantile)
                            )
                    if sample_identity:
                        identity_array = np.asarray(sample_identity, dtype=float)
                        record['identity_gap_mc_mean'] = float(np.mean(identity_array))
                        record['identity_gap_mc_std'] = float(np.std(identity_array, ddof=0))
                        for quantile in quantiles:
                            suffix = f"{int(round(quantile * 100)):02d}"
                            record[f"identity_gap_q{suffix}"] = float(
                                np.quantile(identity_array, quantile)
                            )
                    if sample_net_income:
                        income_array = np.asarray(sample_net_income, dtype=float)
                        record['pred_net_income_mc_mean'] = float(np.mean(income_array))
                        record['pred_net_income_mc_std'] = float(np.std(income_array, ddof=0))
                        for quantile in quantiles:
                            suffix = f"{int(round(quantile * 100)):02d}"
                            record[f"pred_net_income_q{suffix}"] = float(
                                np.quantile(income_array, quantile)
                            )

                records.append(record)
                current_state = pred_state
                seq_window.append(preds_mean_scaled.astype(np.float32))
                if len(seq_window) > sequence_length:
                    seq_window.pop(0)

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
