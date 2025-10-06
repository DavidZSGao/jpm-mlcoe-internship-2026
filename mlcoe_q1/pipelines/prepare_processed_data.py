"""CLI to convert raw statement bundles into processed parquet tables."""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Sequence

from mlcoe_q1.data.statement_processing import save_bundle
from mlcoe_q1.data.yfinance_ingest import StatementFetcher, StoragePaths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Optional ticker subset. If omitted, process all cached JSON bundles.",
    )
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "data",
        help="Base directory containing raw/ and processed/ folders",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Disable live downloads; require cached payloads in raw/.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    storage = StoragePaths(args.data_root)
    fetcher = StatementFetcher(storage)

    if args.cache_only:
        logging.info("Cache-only requested: disabling online downloads")
        fetcher.disable_online()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = sorted(p.stem.upper() for p in storage.raw_dir.glob("*.json"))
        logging.info("No tickers supplied; discovered %d cached bundles", len(tickers))

    for ticker in tickers:
        logging.info("Processing %s", ticker)
        bundle = fetcher.fetch(ticker)
        output_path = save_bundle(bundle, storage)
        if output_path is None:
            logging.warning('No processed parquet generated for %s', ticker)
        else:
            logging.info('Processed parquet written to %s', output_path)


if __name__ == "__main__":  # pragma: no cover
    main()
