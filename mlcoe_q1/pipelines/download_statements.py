"""CLI entry point for downloading or loading cached financial statements."""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Sequence

from mlcoe_q1.data.yfinance_ingest import StatementFetcher, StoragePaths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbols to ingest (e.g., GM JPM MSFT)",
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
        help="Disable live download attempts; require cached payloads.",
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

    logging.info('Data root: %s', storage.root)
    logging.info('Online download enabled: %s', fetcher.online_enabled)

    if args.cache_only:
        logging.info('Cache-only requested: disabling online downloads')
        fetcher.disable_online()

    for ticker in args.tickers:
        logging.info("Processing %s", ticker)
        bundle = fetcher.fetch(ticker)
        logging.info(
            "Loaded statements for %s with %d balance sheet items and %d income items",
            bundle.ticker,
            len(bundle.balance_sheet),
            len(bundle.income_statement),
        )


if __name__ == "__main__":  # pragma: no cover
    main()
