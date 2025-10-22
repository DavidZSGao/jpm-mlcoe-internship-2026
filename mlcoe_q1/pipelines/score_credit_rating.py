"""Score credit ratings using Altman-style features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from mlcoe_q1.credit import AltmanInputs, assign_rating_bucket, compute_altman_z
from mlcoe_q1.pipelines.build_credit_rating_dataset import build_credit_dataset

REQUIRED_KEYS = {
    "total_assets",
    "total_liabilities",
    "current_assets",
    "current_liabilities",
    "retained_earnings",
    "ebit",
    "revenue",
}


def _load_manual_inputs(path: Path) -> AltmanInputs:
    payload = json.loads(path.read_text())
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Manual metrics missing keys: {sorted(missing)}")

    market_equity = payload.get("market_equity") or payload.get("book_equity")
    if market_equity is None:
        raise ValueError("Manual metrics require either 'market_equity' or 'book_equity'")

    return AltmanInputs(
        total_assets=float(payload["total_assets"]),
        total_liabilities=float(payload["total_liabilities"]),
        current_assets=float(payload["current_assets"]),
        current_liabilities=float(payload["current_liabilities"]),
        retained_earnings=float(payload["retained_earnings"]),
        ebit=float(payload["ebit"]),
        revenue=float(payload["revenue"]),
        market_equity=float(market_equity),
    )


def _score_manual(inputs: AltmanInputs, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = compute_altman_z(inputs)
    output: Dict[str, Any] = {
        "z_score": result.z_score,
        "rating_bucket": assign_rating_bucket(result.z_score),
        "working_capital_ratio": result.working_capital_ratio,
        "retained_earnings_ratio": result.retained_earnings_ratio,
        "ebit_ratio": result.ebit_ratio,
        "market_equity_ratio": result.market_equity_ratio,
        "revenue_ratio": result.revenue_ratio,
    }
    if metadata:
        output.update(metadata)
    return output


def _score_ticker(ticker: str, period: Optional[str], min_year: Optional[int]) -> Dict[str, Any]:
    dataset = build_credit_dataset([ticker], min_year=min_year)
    if dataset.empty:
        raise ValueError(f"No financial data available for ticker {ticker}")
    dataset = dataset.copy()
    dataset["period_ts"] = pd.to_datetime(dataset["period"], utc=True)
    if period:
        target_date = pd.to_datetime(period).date()
        match = dataset.loc[dataset["period_ts"].dt.date == target_date]
        if match.empty:
            raise ValueError(f"Requested period {period} unavailable for {ticker}")
        row = match.sort_values("period_ts").iloc[-1]
    else:
        row = dataset.sort_values("period_ts").iloc[-1]
    output = row.drop(labels=["period_ts"]).to_dict()
    output.setdefault("ticker", ticker)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Score credit ratings using Altman features")
    parser.add_argument("--ticker", help="Ticker symbol to score via Yahoo Finance")
    parser.add_argument("--period", help="Reporting period (YYYY-MM-DD) to score")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="JSON file containing manual Altman inputs derived from an annual report",
    )
    parser.add_argument("--company", help="Optional company name for manual inputs")
    parser.add_argument("--min-year", type=int, default=2019)
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON file for the rating summary",
    )
    args = parser.parse_args()

    if bool(args.ticker) == bool(args.metrics_file):
        raise ValueError("Provide either --ticker or --metrics-file, but not both")

    if args.metrics_file:
        inputs = _load_manual_inputs(args.metrics_file)
        metadata = {"company": args.company} if args.company else None
        summary = _score_manual(inputs, metadata)
    else:
        summary = _score_ticker(args.ticker, args.period, args.min_year)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
