"""CLI to derive driver ratios from processed financial statements."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlcoe_q1.utils.driver_features import compute_drivers


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
        "--log-level",
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


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
    features = features.dropna()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    logging.info("Driver dataset saved to %s", args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
