"""Utilities for sourcing balance sheet and income statement data.

The helpers here treat Yahoo Finance as the default backend when
available, but they also support reading pre-downloaded JSON/CSV blobs
from the local `data/raw` folder so that the rest of the pipeline can
function without live network access. The goal is to provide a unified
interface regardless of how the data was obtained.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    from yfinance import const as _yf_const  # type: ignore
except Exception:  # pragma: no cover - yfinance optional
    _yf_const = None

LOGGER = logging.getLogger(__name__)

_YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class StoragePaths:
    """Paths for persisting raw and processed statement payloads."""

    root: pathlib.Path
    raw_dirname: str = "raw"
    processed_dirname: str = "processed"

    @property
    def raw_dir(self) -> pathlib.Path:
        return self.root / self.raw_dirname

    @property
    def processed_dir(self) -> pathlib.Path:
        return self.root / self.processed_dirname

    def ensure(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class StatementBundle:
    ticker: str
    balance_sheet: Dict[str, Any]
    income_statement: Dict[str, Any]
    cashflow_statement: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "balance_sheet": self.balance_sheet,
            "income_statement": self.income_statement,
            "cashflow_statement": self.cashflow_statement,
        }


class StatementFetcher:
    """Fetches statements via Yahoo Finance when available.

    If the runtime cannot import `yfinance` or a download fails, the
    fetcher falls back to reading `{ticker}.json` from `data/raw`. This
    allows benchmarking and model development to continue in offline or
    firewalled environments.
    """

    def __init__(self, storage: StoragePaths) -> None:
        self._storage = storage
        self._storage.ensure()
        try:
            import yfinance as yf  # type: ignore

            self._yfinance = yf
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.warning("yfinance unavailable, falling back to local payloads: %s", exc)
            self._yfinance = None

    @property
    def online_enabled(self) -> bool:
        return self._yfinance is not None

    def disable_online(self) -> None:
        self._yfinance = None

    def fetch(self, ticker: str) -> StatementBundle:
        ticker = ticker.upper()
        if self._yfinance is not None:
            try:
                return self._download_via_yf(ticker)
            except Exception as exc:  # pragma: no cover - network/third-party issues
                LOGGER.error("Yahoo Finance download failed for %s: %s", ticker, exc)

        local_payload = self._load_local_payload(ticker)
        if local_payload is None:
            raise RuntimeError(
                f"No financial statement data available for {ticker}. "
                "Either install yfinance / enable network access or "
                "drop a JSON payload under data/raw/."
            )
        return local_payload

    # ------------------------------------------------------------------
    # Internal helpers
    def _download_via_yf(self, ticker: str) -> StatementBundle:
        assert self._yfinance is not None  # for static checkers
        LOGGER.info("Downloading financial statements for %s via yfinance", ticker)
        ticker_handle = self._yfinance.Ticker(ticker)

        balance = ticker_handle.balance_sheet
        income = ticker_handle.financials
        cashflow = getattr(ticker_handle, "cashflow", None)

        if _frame_is_empty(balance) and _frame_is_empty(income):
            LOGGER.warning("yfinance returned empty frames for %s; attempting direct API fallback", ticker)
            bundle = self._download_via_timeseries(ticker)
            self._persist_raw(bundle)
            return bundle

        bundle = StatementBundle(
            ticker=ticker,
            balance_sheet=_dataframe_to_dict(balance),
            income_statement=_dataframe_to_dict(income),
            cashflow_statement=_dataframe_to_dict(cashflow) if cashflow is not None else None,
        )
        self._persist_raw(bundle)
        return bundle

    def _download_via_timeseries(self, ticker: str) -> StatementBundle:
        LOGGER.info("Fetching %s fundamentals via timeseries fallback", ticker)
        if _yf_const is None:
            raise RuntimeError("yfinance constants unavailable for fallback retrieval")

        session = requests.Session()
        session.headers.update(_YAHOO_HEADERS)

        session.get("https://fc.yahoo.com", timeout=30)
        crumb_resp = session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=30)
        crumb_resp.raise_for_status()
        crumb = crumb_resp.text.strip()
        if not crumb or "Too Many Requests" in crumb:
            raise RuntimeError("Failed to obtain Yahoo crumb token")

        balance_sheet = _timeseries_statement(
            session=session,
            ticker=ticker,
            crumb=crumb,
            statement_key="balance-sheet",
        )
        income_statement = _timeseries_statement(
            session=session,
            ticker=ticker,
            crumb=crumb,
            statement_key="financials",
        )
        cashflow_statement = _timeseries_statement(
            session=session,
            ticker=ticker,
            crumb=crumb,
            statement_key="cash-flow",
        )

        if not balance_sheet and not income_statement and not cashflow_statement:
            raise RuntimeError("Timeseries fallback produced empty statements")

        return StatementBundle(
            ticker=ticker,
            balance_sheet=balance_sheet,
            income_statement=income_statement,
            cashflow_statement=cashflow_statement or None,
        )

    def _persist_raw(self, bundle: StatementBundle) -> None:
        output_path = self._storage.raw_dir / f"{bundle.ticker}.json"
        output_path.write_text(json.dumps(bundle.to_json(), indent=2))
        LOGGER.info("Saved raw statements for %s to %s", bundle.ticker, output_path)

    def _load_local_payload(self, ticker: str) -> Optional[StatementBundle]:
        candidate = self._storage.raw_dir / f"{ticker}.json"
        if not candidate.exists():
            LOGGER.debug("No local payload at %s", candidate)
            return None
        LOGGER.info("Loading cached statements for %s from %s", ticker, candidate)
        payload = json.loads(candidate.read_text())
        return StatementBundle(
            ticker=payload["ticker"],
            balance_sheet=payload.get("balance_sheet", {}),
            income_statement=payload.get("income_statement", {}),
            cashflow_statement=payload.get("cashflow_statement"),
        )



_FUNDAMENTALS_PERIOD1 = int(_dt.datetime(2016, 12, 31).timestamp())
_TYPE_CHUNK = 32


def _frame_is_empty(df: Any) -> bool:
    if df is None:
        return True
    try:
        return df.empty  # type: ignore[no-any-return]
    except AttributeError:
        return False


def _timeseries_statement(*, session: requests.Session, ticker: str, crumb: str, statement_key: str) -> Dict[str, Dict[str, float]]:
    if _yf_const is None:
        return {}
    keys = _yf_const.fundamentals_keys.get(statement_key, [])
    if not keys:
        return {}

    end_ts = int(_dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    url = f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}"

    aggregated: Dict[str, Dict[str, float]] = {}
    for chunk in _chunks(keys, _TYPE_CHUNK):
        types = ','.join(f"annual{k}" for k in chunk)
        params = {
            'type': types,
            'period1': _FUNDAMENTALS_PERIOD1,
            'period2': end_ts,
            'crumb': crumb,
        }
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        results = body.get('timeseries', {}).get('result') or []
        partial = _timeseries_to_dict(results)
        _merge_statement_series(aggregated, partial)
    return aggregated


def _chunks(seq: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(seq), size):
        yield seq[idx: idx + size]



def _timeseries_to_dict(nodes: Any) -> Dict[str, Dict[str, float]]:
    mapping: Dict[str, Dict[str, float]] = {}
    if not isinstance(nodes, list):
        return mapping

    for node in nodes:
        if not isinstance(node, dict):
            continue
        for key, values in node.items():
            if not isinstance(key, str) or not key.startswith('annual'):
                continue
            series_key = _canonicalise_key(key[len('annual'):])
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                period = _extract_period(item.get('asOfDate'))
                if period is None:
                    continue
                numeric = _extract_numeric(item.get('reportedValue'))
                if numeric is None:
                    continue
                mapping.setdefault(series_key, {})[period] = numeric
    return mapping


def _merge_statement_series(target: Dict[str, Dict[str, float]], update: Dict[str, Dict[str, float]]) -> None:
    for key, series in update.items():
        bucket = target.setdefault(key, {})
        bucket.update(series)


def _extract_period(node: Any) -> Optional[str]:
    if isinstance(node, dict):
        if node.get('fmt'):
            return node['fmt']
        node = node.get('raw')
    if isinstance(node, (int, float)):
        try:
            return _dt.datetime.utcfromtimestamp(int(node)).date().isoformat()
        except (ValueError, OSError):
            return None
    if isinstance(node, str) and node:
        return node
    return None


def _extract_numeric(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get('raw')
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _canonicalise_key(name: str) -> str:
    if not name:
        return name
    return name[0].lower() + name[1:]




def _json_serialisable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_json_serialisable(v) for v in value]
    return value


def bulk_fetch(
    tickers: Iterable[str],
    storage_root: pathlib.Path,
) -> List[StatementBundle]:
    """Fetch statements for a batch of tickers and persist raw payloads.

    Parameters
    ----------
    tickers: iterable of ticker strings
        Company tickers to download or load from cache.
    storage_root: pathlib.Path
        Directory containing the `raw/` and `processed/` folders.
    """

    storage = StoragePaths(storage_root)
    fetcher = StatementFetcher(storage)
    results: List[StatementBundle] = []
    for ticker in tickers:
        try:
            results.append(fetcher.fetch(ticker))
        except Exception as exc:
            LOGGER.error("Failed to ingest %s: %s", ticker, exc)
    return results


__all__ = [
    "StatementBundle",
    "StatementFetcher",
    "StoragePaths",
    "bulk_fetch",
]
