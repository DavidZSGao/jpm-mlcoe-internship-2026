"""Helpers for loading processed financial statements into tidy dataframes."""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Iterable, Dict


def load_processed_statement(path: Path) -> pd.DataFrame:
    """Load a single processed parquet file into a multi-indexed dataframe."""

    df = pd.read_parquet(path)
    df['period'] = pd.to_datetime(df['period'])
    return df


def wide_pivot(df: pd.DataFrame, statement: str) -> pd.DataFrame:
    subset = df[df['statement'] == statement]
    return subset.pivot(index='period', columns='line_item', values='value').sort_index()


def load_all_processed(root: Path, tickers: Iterable[str]) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = root / f"{ticker.upper()}.parquet"
        if path.exists():
            result[ticker.upper()] = load_processed_statement(path)
    return result

__all__ = ["load_processed_statement", "wide_pivot", "load_all_processed"]
