from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlcoe_q1.pipelines.orchestrate_lending_workflow import (
    WorkflowOutputs,
    _parse_artifact_override,
    parse_args,
    run_workflow,
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


def test_run_workflow_end_to_end(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "evaluation.parquet"
    _sample_evaluation_dataframe().to_parquet(evaluation_path, index=False)

    outputs = run_workflow(
        evaluation_path=evaluation_path,
        summary_output=tmp_path / "summary.parquet",
        summary_group_cols=["ticker"],
        scenario_output=tmp_path / "scenarios.parquet",
        scenario_spec="baseline:0.5,stress:0.1",
        reasonableness_output=tmp_path / "reason.parquet",
        reasonableness_group_cols=["scenario"],
        macro_config=None,
        macro_output=tmp_path / "macro.parquet",
        calibration_output=tmp_path / "calibration.parquet",
        calibration_group_cols=["ticker"],
        calibration_metrics=["assets", "equity"],
        briefing_output=tmp_path / "briefing.md",
        coverage_target=0.8,
        identity_threshold=1e6,
        package_dir=tmp_path / "package",
        skip_package=False,
        copy_package_artifacts=False,
        artifact_overrides=["extra=/tmp/example.txt"],
    )

    assert isinstance(outputs, WorkflowOutputs)
    assert outputs.summary.exists()
    assert outputs.scenarios.exists()
    assert outputs.reasonableness.exists()
    assert outputs.calibration.exists()
    assert outputs.briefing.exists()
    assert "Strategic Lending" in outputs.briefing.read_text()

    summary_df = pd.read_parquet(outputs.summary)
    assert "assets_mae_mean" in summary_df.columns

    scenario_df = pd.read_parquet(outputs.scenarios)
    assert set(scenario_df["scenario"].unique()) == {"baseline", "stress"}

    macro_df = pd.read_parquet(outputs.macro)
    assert "macro_policy_rate" in macro_df.columns

    reason_df = pd.read_parquet(outputs.reasonableness)
    assert "total_assets_mae" in reason_df.columns

    calibration_df = pd.read_parquet(outputs.calibration)
    expected_cols = {col for col in calibration_df.columns if col.endswith("_coverage")}
    assert expected_cols

    assert outputs.manifest is not None
    assert outputs.readme is not None
    manifest = json.loads(outputs.manifest.read_text())
    labels = {entry["label"] for entry in manifest["artifacts"]}
    assert {"raw_evaluation", "strategic_briefing", "extra"}.issubset(labels)


def test_parse_args_default_grouping() -> None:
    args = parse_args([])
    assert args.summary_group_by == "ticker,mode"


def test_parse_args_with_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    payload = {
        "summary_output": str(tmp_path / "summary.parquet"),
        "macro_config": str(tmp_path / "macros.json"),
        "skip_package": True,
        "copy_artifacts": "true",
        "artifact": ["config=/tmp/configured.txt"],
        "log_level": "DEBUG",
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    args = parse_args(["--config", str(config_path), "--artifact", "cli=/tmp/cli.txt"])

    assert args.config == config_path
    assert args.summary_output == tmp_path / "summary.parquet"
    assert args.macro_config == tmp_path / "macros.json"
    assert args.skip_package is True
    assert args.copy_artifacts is True
    assert args.artifact == ["config=/tmp/configured.txt", "cli=/tmp/cli.txt"]
    assert args.log_level == "DEBUG"


def test_parse_artifact_override_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "foo.txt"
    label, resolved = _parse_artifact_override(f"example={path}")
    assert label == "example"
    assert resolved == path

