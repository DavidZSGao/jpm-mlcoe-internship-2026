"""Generate Altman-style credit rating features from Yahoo Finance statements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from mlcoe_q1.credit import (
    AltmanInputs,
    AltmanResult,
    assign_rating_bucket,
    compute_altman_z,
    derive_altman_inputs,
)


DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "BAC", "C", "GM", "HON", "CAT", "UNP", "3333.HK"]


BALANCE_SHEET_EQUITY_CANDIDATES: Dict[str, Iterable[str]] = {
    "shares_outstanding": (
        "Ordinary Shares Number",
        "Share Issued",
        "Basic Average Shares",
        "Diluted Average Shares",
    ),
    "book_equity": (
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
    ),
}


def _lookup_first(series: pd.Series, candidates: Iterable[str]) -> Optional[float]:
    for name in candidates:
        if name in series.index:
            value = series[name]
            if pd.notna(value):
                return float(value)
    return None


def _fetch_period_close(ticker: str, period: pd.Timestamp, window: int = 10) -> Optional[float]:
    start = (period - pd.Timedelta(days=window)).tz_localize(None)
    end = (period + pd.Timedelta(days=window)).tz_localize(None)
    hist = yf.download(ticker, start=start, end=end + pd.Timedelta(days=1), progress=False)
    if hist.empty:
        return None
    hist = hist[~hist.index.duplicated(keep="last")]
    after = hist.loc[hist.index >= period]
    if not after.empty:
        return float(after["Close"].iloc[0])
    return float(hist["Close"].iloc[-1])


def _prepare_frames(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = pd.to_datetime(frame.columns)
    return frame


def _build_row(
    ticker: str,
    period: pd.Timestamp,
    inputs: AltmanInputs,
    result: AltmanResult,
    shares: Optional[float],
    equity_source: str,
) -> Dict[str, float | str]:
    working_capital = inputs.current_assets - inputs.current_liabilities
    leverage = inputs.total_liabilities / inputs.total_assets
    row: Dict[str, float | str] = {
        "ticker": ticker,
        "period": period.isoformat(),
        "rating_bucket": assign_rating_bucket(result.z_score),
        "z_score": result.z_score,
        "working_capital_ratio": result.working_capital_ratio,
        "retained_earnings_ratio": result.retained_earnings_ratio,
        "ebit_ratio": result.ebit_ratio,
        "market_equity_ratio": result.market_equity_ratio,
        "revenue_ratio": result.revenue_ratio,
        "working_capital": working_capital,
        "total_assets": inputs.total_assets,
        "total_liabilities": inputs.total_liabilities,
        "retained_earnings": inputs.retained_earnings,
        "ebit": inputs.ebit,
        "revenue": inputs.revenue,
        "market_equity": inputs.market_equity,
        "shares_outstanding": shares,
        "leverage": leverage,
        "equity_source": equity_source,
    }

    return row


def _load_statements(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ticker_obj = yf.Ticker(ticker)
    return _prepare_frames(ticker_obj.balance_sheet), _prepare_frames(ticker_obj.income_stmt)


def build_credit_dataset(
    tickers: Iterable[str],
    *,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    statement_loader: Callable[[str], Tuple[pd.DataFrame, pd.DataFrame]] = _load_statements,
    price_fetcher: Callable[[str, pd.Timestamp], Optional[float]] = _fetch_period_close,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []

    for ticker in tickers:
        try:
            balance_sheet, income_statement = statement_loader(ticker)
        except Exception:
            continue
        if balance_sheet.empty or income_statement.empty:
            continue

        periods = sorted(set(balance_sheet.columns) & set(income_statement.columns))
        for period in periods:
            if min_year is not None and period.year < min_year:
                continue
            if max_year is not None and period.year > max_year:
                continue

            bs_slice = balance_sheet[period]
            shares = _lookup_first(bs_slice, BALANCE_SHEET_EQUITY_CANDIDATES["shares_outstanding"])
            book_equity = _lookup_first(bs_slice, BALANCE_SHEET_EQUITY_CANDIDATES["book_equity"])

            price = price_fetcher(ticker, period)
            market_equity = None
            equity_source = "book"
            if price is not None and shares:
                market_equity = price * shares
                equity_source = "market"

            inputs = derive_altman_inputs(
                balance_sheet,
                income_statement,
                period=period,
                market_equity=market_equity,
                fallback_equity=book_equity,
            )
            if inputs is None:
                continue

            result = compute_altman_z(inputs)
            rows.append(
                _build_row(
                    ticker,
                    period,
                    inputs,
                    result,
                    shares,
                    equity_source,
                )
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Altman-style credit rating dataset")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=DEFAULT_TICKERS,
        help="Tickers to include (defaults to internal lending portfolio and Evergrande).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/credit_ratings/altman_features.parquet"),
        help="Destination parquet path for the feature dataset.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/credit_ratings/altman_features.json"),
        help="Optional JSON metadata output summarising dataset coverage.",
    )
    parser.add_argument("--min-year", type=int, default=2019)
    parser.add_argument("--max-year", type=int, default=None)
    args = parser.parse_args()

    dataset = build_credit_dataset(
        args.tickers,
        min_year=args.min_year,
        max_year=args.max_year,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output, index=False)

    metadata = {
        "tickers": sorted(dataset["ticker"].unique()),
        "period_start": dataset["period"].min() if not dataset.empty else None,
        "period_end": dataset["period"].max() if not dataset.empty else None,
        "rows": len(dataset),
        "source": "yfinance",
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2))

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
