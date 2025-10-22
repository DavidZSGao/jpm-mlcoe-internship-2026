"""CLI to derive driver ratios from processed financial statements."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config
from mlcoe_q1.utils.driver_features import (
    BASE_FEATURES,
    OPTIONAL_FEATURES,
    augment_with_lagged_features,
    compute_drivers,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed",
        help="Directory containing processed parquet statements",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Optional subset of tickers (defaults to all parquet files)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/processed/driver_features.parquet",
        help="Path to write the derived dataset",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=0,
        help="Number of trailing periods to append as lagged features",
    )
    parser.add_argument(
        "--lag-features",
        nargs="*",
        help="Specific feature columns to lag (defaults to BASE features if omitted)",
    )
    parser.add_argument(
        "--keep-missing-lags",
        action="store_true",
        help="Retain rows with incomplete lag history instead of dropping them",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity",
    )
    add_config_argument(parser)
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={"tickers": [], "lag_features": []},
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    processed_root = args.data_root
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = [p.stem.upper() for p in processed_root.glob('*.parquet') if p.stem.lower() != 'driver_features']

    logging.info("Computing driver features for %d tickers", len(tickers))
    features = compute_drivers(processed_root, tickers)

    if args.lags:
        lag_columns = args.lag_features or BASE_FEATURES
        logging.info(
            "Adding %d period(s) of lagged features for %d columns",
            args.lags,
            len(lag_columns),
        )
        features = augment_with_lagged_features(
            features,
            lag_columns,
            args.lags,
            drop_missing=not args.keep_missing_lags,
        )

    # Ensure all expected columns are present before enforcing base feature completeness
    for column in OPTIONAL_FEATURES:
        if column not in features.columns:
            features[column] = 0.0
        else:
            features[column] = features[column].fillna(0.0)

    missing_base = [col for col in BASE_FEATURES if col not in features.columns]
    if missing_base:
        raise KeyError(f"Missing required driver features: {missing_base}")

    features = features.dropna(subset=BASE_FEATURES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    logging.info("Driver dataset saved to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
