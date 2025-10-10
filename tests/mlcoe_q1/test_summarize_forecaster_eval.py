"""Tests for the forecaster evaluation summarizer CLI."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from mlcoe_q1.pipelines import summarize_forecaster_eval as mod


@pytest.fixture()
def sample_evaluation_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "JPM", "mode": "mlp", "assets_mae": 1.0, "equity_mae": 2.0, "identity_gap": 0.1},
            {"ticker": "JPM", "mode": "mlp", "assets_mae": 2.0, "equity_mae": 3.0, "identity_gap": 0.2},
            {"ticker": "BAC", "mode": "template", "assets_mae": 4.0, "equity_mae": 5.0, "identity_gap": 0.3},
        ]
    )


def test_validate_columns_raises_for_missing(sample_evaluation_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        mod._validate_columns(sample_evaluation_df.drop(columns=["assets_mae"]), ["assets_mae"])


def test_summarize_groups_and_computes_statistics(sample_evaluation_df: pd.DataFrame) -> None:
    summary = mod.summarize_metrics(sample_evaluation_df, ["ticker", "mode"])

    assert summary.shape == (2, 10)
    jpm = summary.loc[(summary["ticker"] == "JPM") & (summary["mode"] == "mlp")].iloc[0]
    assert jpm["observations"] == 2
    assert jpm["assets_mae_mean"] == pytest.approx(1.5)
    assert jpm["assets_mae_median"] == pytest.approx(1.5)
    assert jpm["assets_mae_max"] == pytest.approx(2.0)
    assert jpm["equity_mae_mean"] == pytest.approx(2.5)
    assert jpm["equity_mae_median"] == pytest.approx(2.5)
    assert jpm["equity_mae_max"] == pytest.approx(3.0)
    assert jpm["identity_gap_mean"] == pytest.approx(0.15)


def test_summarize_includes_net_income_metrics(sample_evaluation_df: pd.DataFrame) -> None:
    df = sample_evaluation_df.copy()
    df["net_income_mae"] = [1.0, 3.0, 2.0]

    summary = mod.summarize_metrics(df, ["ticker", "mode"])
    assert {"net_income_mae_mean", "net_income_mae_median", "net_income_mae_max"}.issubset(summary.columns)

    jpm = summary.loc[(summary["ticker"] == "JPM") & (summary["mode"] == "mlp")].iloc[0]
    assert jpm["net_income_mae_mean"] == pytest.approx(2.0)
    assert jpm["net_income_mae_median"] == pytest.approx(2.0)
    assert jpm["net_income_mae_max"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "suffix, loader",
    [
        (".csv", pd.read_csv),
        (".json", lambda path: pd.read_json(path, orient="records")),
        (".parquet", pd.read_parquet),
    ],
)
def test_write_output_persists_expected_format(
    tmp_path: Path, sample_evaluation_df: pd.DataFrame, suffix: str, loader
) -> None:
    path = tmp_path / f"summary{suffix}"
    mod._write_output(sample_evaluation_df, path)
    loaded = loader(path)
    pd.testing.assert_frame_equal(loaded, sample_evaluation_df, check_dtype=False)


def test_write_output_rejects_unknown_suffix(tmp_path: Path, sample_evaluation_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        mod._write_output(sample_evaluation_df, tmp_path / "summary.txt")


def test_main_runs_end_to_end(tmp_path: Path, sample_evaluation_df: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    evaluation_path = tmp_path / "eval.parquet"
    sample_evaluation_df.to_parquet(evaluation_path, index=False)

    output_path = tmp_path / "summary.csv"
    caplog.set_level(logging.INFO)

    mod.main(["--evaluation", str(evaluation_path), "--output", str(output_path)])

    saved = pd.read_csv(output_path)
    summary = mod.summarize_metrics(sample_evaluation_df, ["ticker", "mode"])
    pd.testing.assert_frame_equal(saved, summary, check_dtype=False)
    assert "Summarised" in caplog.text


def test_main_raises_when_evaluation_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        mod.main(["--evaluation", str(missing)])
