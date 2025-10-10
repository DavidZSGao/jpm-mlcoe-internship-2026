"""Calibrate linear ensembles blending bank templates with neural forecasts."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import tensorflow as tf

from mlcoe_q1.models import tf_forecaster as _tf_forecaster  # noqa: F401 register custom layers
from mlcoe_q1.models.bank_template import deserialize_templates, BankTemplate
from mlcoe_q1.models.bank_ensemble import fit_ensemble_weights, serialize_ensemble
from mlcoe_q1.models.balance_sheet_constraints import project_forward
from mlcoe_q1.pipelines.evaluate_forecaster import (
    _features_to_driver_series,
    row_to_driver,
)
from mlcoe_q1.utils.state_extractor import extract_states

BANK_TICKERS = {"JPM", "BAC", "C"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drivers",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/driver_features.parquet",
        help="Driver feature parquet used during training",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/artifacts/driver_forecaster",
        help="Directory containing the trained model and scaling metadata",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed",
        help="Root directory containing processed statement parquet files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models/artifacts/driver_forecaster/bank_ensemble.json",
        help="Path to persist calibrated ensemble weights",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def gather_bank_records(
    df: pd.DataFrame,
    feature_columns: list[str],
    transform_config: dict[str, str],
    model: tf.keras.Model,
    stats: dict,
    bank_templates: dict[str, BankTemplate],
    processed_root: Path,
) -> dict[str, list[dict[str, float]]]:
    df = df.copy()
    lag_columns = [col for col in df.columns if col.endswith(tuple(f"_lag{i}" for i in range(1, 10)))]
    if lag_columns:
        df[lag_columns] = df[lag_columns].fillna(0.0)
    x_mean = pd.Series(stats["x_mean"], index=feature_columns, dtype=float)
    x_std = pd.Series(stats["x_std"], index=feature_columns, dtype=float)
    x_mean_array = x_mean.to_numpy()
    x_std_array = x_std.to_numpy()
    aux_features: list[str] = stats.get("aux_features", [])
    aux_dim = int(stats.get("aux_dim", len(aux_features)))
    bank_tickers = set(stats.get("bank_tickers", list(BANK_TICKERS)))
    y_std = np.asarray(stats["y_std"], dtype=float)
    y_mean = np.asarray(stats["y_mean"], dtype=float)

    records: dict[str, list[dict[str, float]]] = defaultdict(list)

    for ticker, group in df.groupby("ticker"):
        upper = ticker.upper()
        if upper not in bank_tickers:
            continue

        periods = pd.to_datetime(group["period"]).tolist()
        if len(periods) < 2:
            continue

        features = group[feature_columns].astype(float).to_numpy()

        sector_vec = np.zeros(aux_dim, dtype=float)
        for idx, feature in enumerate(aux_features):
            if feature == "is_bank":
                sector_vec[idx] = 1.0

        processed_path = processed_root / f"{ticker}.parquet"
        states = extract_states(processed_path)
        template = bank_templates.get(upper)
        if template is None:
            continue

        for i in range(len(features) - 1):
            prev_period = periods[i]
            target_period = periods[i + 1]
            prev_state = states.get(prev_period)
            target_state = states.get(target_period)
            if prev_state is None or target_state is None:
                continue

            features_prev = features[i]

            scaled = (features_prev - x_mean_array) / x_std_array
            inputs = np.concatenate([scaled, sector_vec], axis=0).reshape(1, -1)
            preds_scaled = model.predict(inputs, verbose=0)
            preds = preds_scaled * y_std + y_mean
            pred_map = {feature_columns[j]: float(preds[0, j]) for j in range(len(feature_columns))}
            driver_series = _features_to_driver_series(pred_map, transform_config)
            if driver_series.isna().any():
                logging.debug(
                    "Skipping %s %s due to missing predicted drivers", ticker, target_period
                )
                continue
            driver_pred = row_to_driver(driver_series)

            template_result = template.project(prev_state)
            mlp_result = project_forward(prev_state, driver_pred)

            target_assets = target_state.total_assets()
            target_equity = target_state.equity

            records[upper].append(
                {
                    "template_assets": template_result.state.total_assets(),
                    "mlp_assets": mlp_result.state.total_assets(),
                    "true_assets": target_assets,
                    "template_equity": template_result.state.equity,
                    "mlp_equity": mlp_result.state.equity,
                    "true_equity": target_equity,
                }
            )

    return records


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    df = pd.read_parquet(args.drivers)
    model = tf.keras.models.load_model(args.model_dir / "model.keras")

    stats_path = args.model_dir / "scaling.json"
    if not stats_path.exists():
        raise FileNotFoundError("Missing scaling.json in model directory")
    with open(stats_path) as fh:
        stats = json.load(fh)

    feature_columns: list[str] = stats.get("feature_columns", [])
    transform_config: dict[str, str] = stats.get("transform_config", {})

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Driver dataset missing columns: {missing}")

    template_path = args.model_dir / "bank_templates.json"
    if not template_path.exists():
        raise FileNotFoundError("Bank templates not found; run training pipeline first")
    with open(template_path) as fh:
        template_payload = json.load(fh)
    bank_templates = deserialize_templates(template_payload)

    records = gather_bank_records(
        df=df,
        feature_columns=feature_columns,
        transform_config=transform_config,
        model=model,
        stats=stats,
        bank_templates=bank_templates,
        processed_root=args.processed_root,
    )

    if not records:
        raise RuntimeError("No bank records found for calibration")

    weights = []
    for ticker, ticker_records in records.items():
        if len(ticker_records) < 2:
            logging.warning("Skipping %s due to insufficient history", ticker)
            continue
        weights.append(fit_ensemble_weights(ticker_records, ticker))

    if not weights:
        raise RuntimeError("Failed to calibrate any bank ensembles")

    payload = serialize_ensemble(weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)

    logging.info("Calibrated ensembles written to %s", args.output)


if __name__ == "__main__":
    main()

