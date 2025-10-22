from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlcoe_q1.pipelines import generate_executive_summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_executive_summary_builds_markdown(tmp_path: Path) -> None:
    forecaster_path = tmp_path / "forecaster.parquet"
    forecaster_df = pd.DataFrame(
        [
            {
                "ticker": "GM",
                "mode": "ensemble",
                "observations": 2,
                "assets_mae_mean": 0.8,
                "equity_mae_mean": 0.4,
                "net_income_mae_mean": 0.2,
                "identity_gap_mean": 0.01,
                "assets_interval_coverage": 0.9,
            },
            {
                "ticker": "AAPL",
                "mode": "ensemble",
                "observations": 1,
                "assets_mae_mean": 1.2,
                "equity_mae_mean": 0.6,
                "net_income_mae_mean": 0.3,
                "identity_gap_mean": 0.02,
                "assets_interval_coverage": 0.85,
            },
        ]
    )
    forecaster_df.to_parquet(forecaster_path, index=False)

    scenario_path = tmp_path / "scenarios.parquet"
    scenario_df = pd.DataFrame(
        [
            {
                "scenario": "baseline",
                "observations": 3,
                "total_assets_mae": 1.0,
                "net_income_mae": 0.5,
                "identity_gap_mae": 0.02,
                "total_assets_interval_coverage": 0.8,
            },
            {
                "scenario": "downside",
                "observations": 3,
                "total_assets_mae": 1.5,
                "net_income_mae": 0.7,
                "identity_gap_mae": 0.03,
                "total_assets_interval_coverage": 0.9,
            },
        ]
    )
    scenario_df.to_parquet(scenario_path, index=False)

    calibration_path = tmp_path / "calibration.parquet"
    calibration_df = pd.DataFrame(
        [
            {
                "ticker": "GM",
                "observations": 5,
                "assets_q50_abs_error": 0.02,
                "equity_q50_abs_error": 0.01,
            }
        ]
    )
    calibration_df.to_parquet(calibration_path, index=False)

    macro_path = tmp_path / "macro.parquet"
    macro_df = pd.DataFrame(
        [
            {
                "scenario": "baseline",
                "macro_assumptions_json": json.dumps({"gdp_growth": 0.02}),
                "applied_adjustments_json": json.dumps({}),
                "applied_adjustment_count": 0,
            },
            {
                "scenario": "stress",
                "macro_assumptions_json": json.dumps({"gdp_growth": -0.01}),
                "applied_adjustments_json": json.dumps({"pred_net_income": {"mode": "pct"}}),
                "applied_adjustment_count": 1,
            },
        ]
    )
    macro_df.to_parquet(macro_path, index=False)

    llm_path = tmp_path / "llm.parquet"
    llm_df = pd.DataFrame(
        [
            {
                "adapter": "openai-chat",
                "model": "gpt-4o-mini",
                "records": 10,
                "mae_mean": 0.4,
                "mae_std": 0.05,
                "coverage_mean": 0.7,
                "coverage_std": 0.1,
                "seed_count": 3,
            }
        ]
    )
    llm_df.to_parquet(llm_path, index=False)

    loan_path = tmp_path / "loan.json"
    _write_json(
        loan_path,
        {
            "baseline": {"avg_rate": 0.05, "avg_spread": 0.02, "count": 4},
            "stress": {"avg_rate": 0.07, "avg_spread": 0.03, "count": 4},
        },
    )

    credit_path = tmp_path / "credit.json"
    _write_json(
        credit_path,
        {
            "tickers": ["GM", "AAPL"],
            "period_start": "2019-12-31",
            "period_end": "2023-12-31",
            "rows": 12,
        },
    )

    risk_path = tmp_path / "risk.json"
    _write_json(
        risk_path,
        {
            "summary": [
                {"issuer": "GM", "warning_count": 2},
                {"issuer": "AAPL", "warning_count": 1},
            ]
        },
    )

    output_path = tmp_path / "summary.md"

    exit_code = generate_executive_summary.main(
        [
            "--forecaster-summary",
            str(forecaster_path),
            "--scenario-summary",
            str(scenario_path),
            "--calibration-report",
            str(calibration_path),
            "--macro-scenarios",
            str(macro_path),
            "--llm-seed-summary",
            str(llm_path),
            "--loan-pricing-summary",
            str(loan_path),
            "--credit-metadata",
            str(credit_path),
            "--risk-summary",
            str(risk_path),
            "--output",
            str(output_path),
            "--title",
            "Executive Summary Test",
        ]
    )

    assert exit_code == 0
    content = output_path.read_text(encoding="utf-8")
    assert "Executive Summary Test" in content
    assert "Forecast quality" in content
    assert "LLM benchmarking" in content
    assert "Loan pricing" in content
    assert "Risk warnings" in content
    assert "openai-chat/gpt-4o-mini" in content


def test_generate_executive_summary_handles_missing_inputs(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.md"
    exit_code = generate_executive_summary.main(
        [
            "--forecaster-summary",
            str(tmp_path / "missing_forecaster.parquet"),
            "--scenario-summary",
            str(tmp_path / "missing_scenarios.parquet"),
            "--calibration-report",
            str(tmp_path / "missing_calibration.parquet"),
            "--macro-scenarios",
            str(tmp_path / "missing_macro.parquet"),
            "--llm-seed-summary",
            str(tmp_path / "missing_llm.parquet"),
            "--loan-pricing-summary",
            str(tmp_path / "missing_loans.json"),
            "--credit-metadata",
            str(tmp_path / "missing_credit.json"),
            "--risk-summary",
            str(tmp_path / "missing_risk.json"),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    content = output_path.read_text(encoding="utf-8")
    assert "Evaluation summary not found" in content
    assert "Scenario summary not found" in content
    assert "LLM summary not found" in content
