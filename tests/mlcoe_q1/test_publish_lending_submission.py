from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlcoe_q1.pipelines.audit_lending_artifacts import Expectation
from mlcoe_q1.pipelines.publish_lending_submission import (
    PublishOutputs,
    parse_args,
    publish_submission,
)


def _sample_evaluation_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["GM", "GM"],
            "prev_period": ["2022-12-31", "2023-03-31"],
            "target_period": ["2023-03-31", "2023-06-30"],
            "horizon": [1, 1],
            "mode": ["ensemble", "ensemble"],
            "distribution": ["gaussian", "gaussian"],
            "mc_strategy": ["dropout", "dropout"],
            "mc_sample_count": [50, 50],
            "assets_mae": [1.0, 2.0],
            "equity_mae": [0.5, 0.7],
            "identity_gap": [1000.0, -500.0],
            "net_income_mae": [0.2, 0.3],
            "pred_total_assets": [100.0, 110.0],
            "pred_equity": [40.0, 42.0],
            "pred_net_income": [5.0, 5.5],
            "identity_gap_q010": [-200.0, -150.0],
            "identity_gap_q090": [200.0, 175.0],
            "pred_total_assets_q010": [95.0, 105.0],
            "pred_total_assets_q090": [105.0, 115.0],
            "pred_equity_q010": [38.0, 40.5],
            "pred_equity_q090": [42.0, 43.5],
            "pred_net_income_q010": [4.5, 5.0],
            "pred_net_income_q090": [5.5, 6.0],
            "true_total_assets": [98.0, 111.0],
            "true_equity": [39.5, 43.0],
            "true_net_income": [4.8, 5.2],
        }
    )


def _write_optional_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    llm_path = tmp_path / "llm.parquet"
    pd.DataFrame(
        {
            "adapter": ["hf"],
            "model": ["t5-small"],
            "records": [4],
            "mae_mean": [1.2],
            "mae_std": [0.1],
            "coverage_mean": [0.8],
            "coverage_std": [0.05],
            "seed_count": [2],
        }
    ).to_parquet(llm_path, index=False)

    loan_path = tmp_path / "loan.json"
    loan_payload = {
        "baseline": {"avg_rate": 0.045, "avg_spread": 0.02, "count": 3}
    }
    loan_path.write_text(json.dumps(loan_payload), encoding="utf-8")

    credit_path = tmp_path / "credit.json"
    credit_payload = {
        "tickers": ["GM", "HON"],
        "period_start": "2018-12-31",
        "period_end": "2023-12-31",
        "rows": 20,
    }
    credit_path.write_text(json.dumps(credit_payload), encoding="utf-8")

    risk_path = tmp_path / "risk.json"
    risk_payload = {
        "GM": {"liquidity": ["tight working capital"]},
    }
    risk_path.write_text(json.dumps(risk_payload), encoding="utf-8")

    return llm_path, loan_path, credit_path, risk_path


def test_publish_submission_end_to_end(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.parquet"
    _sample_evaluation_dataframe().to_parquet(evaluation_path, index=False)

    llm_path, loan_path, credit_path, risk_path = _write_optional_inputs(tmp_path)

    package_dir = tmp_path / "package"
    zip_path = tmp_path / "deliverable.zip"

    outputs = publish_submission(
        evaluation_path=evaluation_path,
        summary_output=tmp_path / "summary.parquet",
        summary_group_cols=["ticker"],
        scenario_output=tmp_path / "scenarios.parquet",
        scenario_spec="baseline:0.5",
        reasonableness_output=tmp_path / "reason.parquet",
        reasonableness_group_cols=["scenario"],
        macro_config=None,
        macro_output=tmp_path / "macro.parquet",
        calibration_output=tmp_path / "calibration.parquet",
        calibration_group_cols=["ticker"],
        calibration_metrics=["assets"],
        briefing_output=tmp_path / "briefing.md",
        coverage_target=0.9,
        identity_threshold=1e6,
        package_dir=package_dir,
        skip_package=False,
        copy_artifacts=False,
        artifact_overrides=[],
        executive_summary_output=tmp_path / "executive.md",
        executive_summary_title="Executive Summary",
        skip_executive_summary=False,
        llm_seed_summary=llm_path,
        loan_pricing_summary=loan_path,
        credit_metadata=credit_path,
        risk_summary=risk_path,
        skip_audit=False,
        audit_expectations=[
            Expectation("Evaluation", tmp_path / "summary.parquet"),
            Expectation("Executive", tmp_path / "executive.md", optional=True),
        ],
        audit_json_output=tmp_path / "audit.json",
        audit_markdown_output=tmp_path / "audit.md",
        zip_output=zip_path,
    )

    assert isinstance(outputs, PublishOutputs)
    assert outputs.workflow.summary.exists()
    assert outputs.executive_summary and outputs.executive_summary.exists()
    assert outputs.audit_json and outputs.audit_json.exists()
    assert outputs.audit_markdown and outputs.audit_markdown.exists()
    assert outputs.package_archive and outputs.package_archive.exists()

    summary_df = pd.read_parquet(outputs.workflow.summary)
    assert "assets_mae_mean" in summary_df.columns

    archive = outputs.package_archive
    assert archive.name.endswith(".zip")


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.summary_group_by == "ticker,mode"
    assert args.skip_audit is False


def test_parse_args_with_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    payload = {
        "summary_output": str(tmp_path / "summary.parquet"),
        "macro_config": str(tmp_path / "macros.json"),
        "skip_package": True,
        "artifact": ["config=/tmp/from_config.txt"],
        "zip_output": str(tmp_path / "bundle.zip"),
        "copy_artifacts": True,
        "log_level": "DEBUG",
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    args = parse_args(["--config", str(config_path), "--artifact", "cli=/tmp/from_cli.txt"])

    assert args.config == config_path
    assert args.summary_output == tmp_path / "summary.parquet"
    assert args.macro_config == tmp_path / "macros.json"
    assert args.skip_package is True
    assert args.copy_artifacts is True
    assert args.artifact == ["config=/tmp/from_config.txt", "cli=/tmp/from_cli.txt"]
    assert args.zip_output == tmp_path / "bundle.zip"
    assert args.log_level == "DEBUG"

