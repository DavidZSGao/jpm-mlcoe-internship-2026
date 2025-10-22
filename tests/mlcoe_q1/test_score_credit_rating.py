from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import mlcoe_q1.pipelines.score_credit_rating as scr
from mlcoe_q1.credit import AltmanInputs


def test_load_manual_inputs_prefers_market_equity(tmp_path: Path):
    payload = {
        "total_assets": 500,
        "total_liabilities": 250,
        "current_assets": 220,
        "current_liabilities": 120,
        "retained_earnings": 80,
        "ebit": 60,
        "revenue": 400,
        "market_equity": 260,
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload))

    inputs = scr._load_manual_inputs(path)
    assert isinstance(inputs, AltmanInputs)
    assert inputs.market_equity == 260


def test_score_manual_outputs_rating():
    inputs = AltmanInputs(
        total_assets=500.0,
        total_liabilities=250.0,
        current_assets=220.0,
        current_liabilities=120.0,
        retained_earnings=80.0,
        ebit=60.0,
        revenue=400.0,
        market_equity=260.0,
    )
    summary = scr._score_manual(inputs, metadata={"company": "Example"})
    assert summary["company"] == "Example"
    assert "rating_bucket" in summary


def test_score_ticker_returns_latest_period(monkeypatch):
    dataset = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period": "2023-12-31T00:00:00",
                "rating_bucket": "bb",
                "z_score": 1.9,
                "market_equity_ratio": 0.3,
            },
            {
                "ticker": "AAA",
                "period": "2024-12-31T00:00:00",
                "rating_bucket": "bbb",
                "z_score": 2.6,
                "market_equity_ratio": 0.4,
            },
        ]
    )

    def stub_build_credit_dataset(tickers, min_year=None, statement_loader=None, price_fetcher=None):
        return dataset

    monkeypatch.setattr(scr, "build_credit_dataset", stub_build_credit_dataset)

    summary = scr._score_ticker("AAA", None, min_year=2019)
    assert summary["period"] == "2024-12-31T00:00:00"
    assert summary["rating_bucket"] == "bbb"


def test_score_ticker_specific_period(monkeypatch):
    dataset = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period": "2023-12-31T00:00:00",
                "rating_bucket": "bb",
                "z_score": 1.9,
            },
            {
                "ticker": "AAA",
                "period": "2024-12-31T00:00:00",
                "rating_bucket": "bbb",
                "z_score": 2.6,
            },
        ]
    )

    def stub_build_credit_dataset(tickers, min_year=None, statement_loader=None, price_fetcher=None):
        return dataset

    monkeypatch.setattr(scr, "build_credit_dataset", stub_build_credit_dataset)

    summary = scr._score_ticker("AAA", "2023-12-31", min_year=2019)
    assert summary["period"] == "2023-12-31T00:00:00"
    assert summary["rating_bucket"] == "bb"


def test_score_ticker_raises_for_missing_period(monkeypatch):
    dataset = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "period": "2024-12-31T00:00:00",
                "rating_bucket": "bbb",
                "z_score": 2.6,
            }
        ]
    )

    def stub_build_credit_dataset(tickers, min_year=None, statement_loader=None, price_fetcher=None):
        return dataset

    monkeypatch.setattr(scr, "build_credit_dataset", stub_build_credit_dataset)

    with pytest.raises(ValueError):
        scr._score_ticker("AAA", "2023-12-31", min_year=2019)
