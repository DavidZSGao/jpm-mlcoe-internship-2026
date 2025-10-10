"""Tests for driver feature engineering utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlcoe_q1.utils.driver_features import (
    OPTIONAL_FEATURES,
    compute_drivers_for_ticker,
    augment_with_lagged_features,
)


def test_bank_optional_features_populated() -> None:
    """Bank tickers should emit non-zero interest-driven ratios."""

    df = compute_drivers_for_ticker(Path("mlcoe_q1/data/processed"), "JPM")
    assert set(OPTIONAL_FEATURES).issubset(df.columns)
    assert (df["net_interest_margin"].abs() > 0).any()
    assert (df["interest_income_ratio"].abs() > 0).any()
    assert (df["interest_expense_ratio"].abs() > 0).any()
    assert (df[["asset_growth", "equity_growth"]].abs() > 0).any().any()


def test_non_bank_optional_features_are_defined() -> None:
    """Non-bank tickers should still expose optional features without NaNs."""

    df = compute_drivers_for_ticker(Path("mlcoe_q1/data/processed"), "CAT")
    assert set(OPTIONAL_FEATURES).issubset(df.columns)
    assert not df[OPTIONAL_FEATURES].isna().any().any()
    # Industrial names should have low-magnitude interest ratios
    assert (df["net_interest_margin"].abs() < 0.05).all()
    assert (df["interest_income_ratio"].abs() < 0.05).all()
    assert (df["interest_expense_ratio"].abs() < 0.05).all()
    assert (df[["asset_growth", "equity_growth", "net_income_growth"]].abs() < 1.5).all().all()


def test_tangible_equity_growth_defaults_to_zero_when_missing() -> None:
    """Tickers without tangible equity should still expose zero-filled growth."""

    df = compute_drivers_for_ticker(Path("mlcoe_q1/data/processed"), "AAPL")
    assert "tangible_equity_growth" in df.columns
    assert not df["tangible_equity_growth"].isna().any()
    assert (df["tangible_equity_growth"].abs() < 5).all()


def test_augment_with_lagged_features_adds_shifted_columns() -> None:
    """Lag augmentation should shift features and drop early periods by default."""

    df = compute_drivers_for_ticker(Path("mlcoe_q1/data/processed"), "GM")
    df = df.sort_values("period").reset_index(drop=True)

    augmented = augment_with_lagged_features(df, ["sales", "sales_growth"], lags=2)
    assert {
        "sales_lag1",
        "sales_lag2",
        "sales_growth_lag1",
        "sales_growth_lag2",
    }.issubset(augmented.columns)

    augmented = augmented.sort_values("period").reset_index(drop=True)
    assert len(augmented) == max(len(df) - 2, 0)

    if len(augmented) >= 1 and len(df) >= 3:
        assert augmented.loc[0, "sales_lag1"] == pytest.approx(df.loc[1, "sales"])
        assert augmented.loc[0, "sales_lag2"] == pytest.approx(df.loc[0, "sales"])
