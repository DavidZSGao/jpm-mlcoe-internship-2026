"""Transform raw statement bundles into tabular datasets."""

from __future__ import annotations

import logging
import pathlib
from typing import Dict, Iterable, List, Optional

import pandas as pd

from mlcoe_q1.data.yfinance_ingest import StatementBundle, StoragePaths

LOGGER = logging.getLogger(__name__)


StatementDict = Dict[str, Dict[str, float]]


def bundle_to_frame(bundle: StatementBundle) -> pd.DataFrame:
    """Flatten a ``StatementBundle`` into a long-form DataFrame.

    Columns:
        ticker, statement, line_item, period, value
    """

    records: List[Dict[str, str | float]] = []

    def _extend(statement: str, payload: StatementDict) -> None:
        for line_item, dated_values in payload.items():
            if not isinstance(dated_values, dict):
                continue
            for period, value in dated_values.items():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                records.append(
                    {
                        "ticker": bundle.ticker,
                        "statement": statement,
                        "line_item": line_item,
                        "period": period,
                        "value": numeric_value,
                    }
                )

    _extend("balance_sheet", bundle.balance_sheet)
    _extend("income_statement", bundle.income_statement)
    if bundle.cashflow_statement:
        _extend("cashflow_statement", bundle.cashflow_statement)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        LOGGER.warning("No tabular data extracted for %s", bundle.ticker)
        return frame

    frame["period"] = pd.to_datetime(frame["period"], errors="coerce")
    frame.sort_values(["statement", "line_item", "period"], inplace=True)
    return frame


def save_bundle(bundle: StatementBundle, storage: StoragePaths) -> Optional[pathlib.Path]:
    """Persist a bundle in parquet form under the processed directory."""

    frame = bundle_to_frame(bundle)
    storage.ensure()
    output = storage.processed_dir / f"{bundle.ticker}.parquet"
    if frame.empty:
        LOGGER.warning("Skipping parquet write for %s; no usable rows", bundle.ticker)
        return None

    frame.to_parquet(output, index=False)
    LOGGER.info("Saved processed dataframe for %s to %s", bundle.ticker, output)
    return output


def bulk_save(bundles: Iterable[StatementBundle], storage: StoragePaths) -> List[pathlib.Path]:
    paths: List[pathlib.Path] = []
    for bundle in bundles:
        path = save_bundle(bundle, storage)
        if path is not None:
            paths.append(path)
    return paths


__all__ = ["bundle_to_frame", "save_bundle", "bulk_save"]
