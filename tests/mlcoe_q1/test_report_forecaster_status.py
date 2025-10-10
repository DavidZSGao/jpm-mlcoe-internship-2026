"""Tests for the forecaster status report CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlcoe_q1.pipelines import report_forecaster_status as mod
from mlcoe_q1.pipelines.summarize_forecaster_eval import summarize_metrics


def _sample_evaluation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "JPM",
                "mode": "bank_template",
                "assets_mae": 1.2e11,
                "equity_mae": 2.5e11,
                "identity_gap": 0.0,
                "net_income_mae": float("nan"),
            },
            {
                "ticker": "AAPL",
                "mode": "mlp",
                "assets_mae": 5.0e9,
                "equity_mae": 1.5e10,
                "identity_gap": 2.0e9,
                "net_income_mae": 3.0e9,
            },
            {
                "ticker": "AAPL",
                "mode": "mlp",
                "assets_mae": 6.0e9,
                "equity_mae": 1.7e10,
                "identity_gap": 2.2e9,
                "net_income_mae": 2.0e9,
            },
        ]
    )


def test_render_report_includes_sections() -> None:
    summary = summarize_metrics(_sample_evaluation(), ["ticker", "mode"])
    report = mod.render_report(summary)

    assert "# Forecaster Status Report" in report
    assert "## Aggregate Metrics" in report
    assert "Highest equity MAE tickers" in report
    assert "Identity gap outliers" in report
    assert "Highest net income MAE tickers" in report


def test_main_writes_report_file(tmp_path: Path, capsys) -> None:
    evaluation_path = tmp_path / "evaluation.parquet"
    _sample_evaluation().to_parquet(evaluation_path, index=False)

    output_path = tmp_path / "status.md"
    mod.main([
        "--evaluation",
        str(evaluation_path),
        "--output",
        str(output_path),
    ])

    contents = output_path.read_text(encoding="utf-8")
    assert "Forecaster Status Report" in contents
    assert "Highest assets MAE tickers" in contents

    std_out = capsys.readouterr().out
    assert "Forecaster Status Report" in std_out

