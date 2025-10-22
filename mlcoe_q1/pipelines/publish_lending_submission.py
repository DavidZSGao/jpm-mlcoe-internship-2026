"""Publish a Strategic Lending submission with orchestration, summary, and audit."""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mlcoe_q1.pipelines import orchestrate_lending_workflow as orchestrator
from mlcoe_q1.pipelines.audit_lending_artifacts import (
    Expectation,
    _default_expectations,
    audit,
    load_expectations,
    report_to_json,
    report_to_markdown,
)
from mlcoe_q1.pipelines.generate_executive_summary import build_summary
from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PublishOutputs:
    """Artifacts produced by the publishing workflow."""

    workflow: orchestrator.WorkflowOutputs
    executive_summary: Path | None
    audit_json: Path | None
    audit_markdown: Path | None
    package_archive: Path | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the publishing workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/forecaster_evaluation.parquet",
        help="Detailed evaluation parquet produced by evaluate_forecaster",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/forecaster_evaluation_summary.parquet",
        help="Destination path for the aggregated evaluation summary",
    )
    parser.add_argument(
        "--summary-group-by",
        type=str,
        default="ticker,mode",
        help="Comma-separated columns for evaluation aggregation",
    )
    parser.add_argument(
        "--scenario-output",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/forecaster_scenarios.parquet",
        help="Destination path for packaged forecast scenarios",
    )
    parser.add_argument(
        "--scenario-spec",
        type=str,
        default="baseline:0.5,downside:0.1,upside:0.9",
        help="Comma separated scenario_name:quantile pairs",
    )
    parser.add_argument(
        "--reasonableness-output",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/scenario_reasonableness.parquet",
        help="Destination for scenario reasonableness diagnostics",
    )
    parser.add_argument(
        "--reasonableness-group-by",
        type=str,
        default="scenario",
        help="Comma separated grouping columns for scenario diagnostics",
    )
    parser.add_argument(
        "--macro-config",
        type=Path,
        default=None,
        help="Optional JSON file describing macro overlays",
    )
    parser.add_argument(
        "--macro-output",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/macro_conditioned_scenarios.parquet",
        help="Destination for macro-conditioned scenarios",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/forecaster_calibration.parquet",
        help="Destination path for calibration diagnostics",
    )
    parser.add_argument(
        "--calibration-group-by",
        type=str,
        default="ticker,mode",
        help="Comma separated grouping columns for calibration metrics",
    )
    parser.add_argument(
        "--calibration-metrics",
        type=str,
        default="assets,equity,net_income",
        help="Comma separated list of calibration metrics to evaluate",
    )
    parser.add_argument(
        "--briefing-output",
        type=Path,
        default=REPO_ROOT / "reports/q1/status/strategic_lending_briefing.md",
        help="Destination Markdown file for the Strategic Lending briefing",
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=0.9,
        help="Target coverage level flagged by the briefing",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=5e8,
        help="Absolute accounting identity threshold for warnings",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=REPO_ROOT / "reports/q1/deliverables",
        help="Directory where manifest/README outputs are written",
    )
    parser.add_argument(
        "--skip-package",
        action="store_true",
        help="Disable manifest/README generation and artifact bundling",
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="Copy artifacts into the package directory",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional artifact overrides for the package manifest",
    )
    parser.add_argument(
        "--executive-summary-output",
        type=Path,
        default=REPO_ROOT / "reports/q1/status/executive_summary.md",
        help="Destination for the consolidated executive summary",
    )
    parser.add_argument(
        "--executive-summary-title",
        type=str,
        default="Strategic Lending Executive Summary",
        help="Heading used inside the executive summary",
    )
    parser.add_argument(
        "--skip-executive-summary",
        action="store_true",
        help="Skip executive summary generation",
    )
    parser.add_argument(
        "--llm-seed-summary",
        type=Path,
        default=REPO_ROOT
        / "reports/q1/artifacts/llm_benchmarks/summary_by_model.parquet",
        help="Seed-aggregated LLM benchmark parquet",
    )
    parser.add_argument(
        "--loan-pricing-summary",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/loan_pricing_summary.json",
        help="Loan pricing summary JSON emitted by price_loans",
    )
    parser.add_argument(
        "--credit-metadata",
        type=Path,
        default=REPO_ROOT / "data/credit_ratings/altman_features.json",
        help="Credit metadata JSON produced by build_credit_rating_dataset",
    )
    parser.add_argument(
        "--risk-summary",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/risk_warnings_summary.json",
        help="Risk warning summary JSON emitted by extract_risk_warnings",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip artifact auditing",
    )
    parser.add_argument(
        "--audit-config",
        type=Path,
        help="Optional JSON file describing artifact expectations",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=REPO_ROOT / "reports/q1/status/lending_artifact_audit.json",
        help="Destination for the JSON audit report",
    )
    parser.add_argument(
        "--audit-markdown-output",
        type=Path,
        help="Optional Markdown destination (defaults to <audit-output>.md)",
    )
    parser.add_argument(
        "--zip-output",
        type=Path,
        help="Optional destination for a zipped deliverable package",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level for status messages",
    )
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "macro_config": Path,
            "audit_config": Path,
            "audit_markdown_output": Path,
            "zip_output": Path,
        },
    )


def _make_archive(path: Path, source_dir: Path) -> Path:
    """Create a zip archive for the provided directory."""

    stem = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(stem), "zip", root_dir=source_dir)
    return Path(archive)


def publish_submission(
    *,
    evaluation_path: Path,
    summary_output: Path,
    summary_group_cols: Sequence[str],
    scenario_output: Path,
    scenario_spec: str,
    reasonableness_output: Path,
    reasonableness_group_cols: Sequence[str],
    macro_config: Path | None,
    macro_output: Path,
    calibration_output: Path,
    calibration_group_cols: Sequence[str],
    calibration_metrics: Sequence[str],
    briefing_output: Path,
    coverage_target: float,
    identity_threshold: float,
    package_dir: Path | None,
    skip_package: bool,
    copy_artifacts: bool,
    artifact_overrides: Sequence[str],
    executive_summary_output: Path | None,
    executive_summary_title: str,
    skip_executive_summary: bool,
    llm_seed_summary: Path | None,
    loan_pricing_summary: Path | None,
    credit_metadata: Path | None,
    risk_summary: Path | None,
    skip_audit: bool,
    audit_expectations: Sequence[Expectation] | None,
    audit_json_output: Path | None,
    audit_markdown_output: Path | None,
    zip_output: Path | None,
) -> PublishOutputs:
    """Execute the Strategic Lending publishing workflow."""

    workflow_outputs = orchestrator.run_workflow(
        evaluation_path=evaluation_path,
        summary_output=summary_output,
        summary_group_cols=summary_group_cols,
        scenario_output=scenario_output,
        scenario_spec=scenario_spec,
        reasonableness_output=reasonableness_output,
        reasonableness_group_cols=reasonableness_group_cols,
        macro_config=macro_config,
        macro_output=macro_output,
        calibration_output=calibration_output,
        calibration_group_cols=calibration_group_cols,
        calibration_metrics=calibration_metrics,
        briefing_output=briefing_output,
        coverage_target=coverage_target,
        identity_threshold=identity_threshold,
        package_dir=package_dir,
        skip_package=skip_package,
        copy_package_artifacts=copy_artifacts,
        artifact_overrides=artifact_overrides,
    )

    summary_path: Path | None = None
    if not skip_executive_summary and executive_summary_output is not None:
        logging.info("Generating executive summary at %s", executive_summary_output)
        args = argparse.Namespace(
            forecaster_summary=workflow_outputs.summary,
            scenario_summary=workflow_outputs.reasonableness,
            calibration_report=workflow_outputs.calibration,
            macro_scenarios=workflow_outputs.macro,
            llm_seed_summary=llm_seed_summary,
            loan_pricing_summary=loan_pricing_summary,
            credit_metadata=credit_metadata,
            risk_summary=risk_summary,
            output=executive_summary_output,
            title=executive_summary_title,
            log_level="INFO",
        )
        content = build_summary(args)
        executive_summary_output.parent.mkdir(parents=True, exist_ok=True)
        executive_summary_output.write_text(content, encoding="utf-8")
        summary_path = executive_summary_output

    audit_json_path: Path | None = None
    audit_markdown_path: Path | None = None
    if not skip_audit:
        expectations = list(audit_expectations) if audit_expectations else _default_expectations()
        logging.info("Auditing %d artifact expectations", len(expectations))
        report = audit(expectations)
        if audit_json_output is not None:
            audit_json_output.parent.mkdir(parents=True, exist_ok=True)
            audit_json_output.write_text(report_to_json(report), encoding="utf-8")
            audit_json_path = audit_json_output
        markdown_path = audit_markdown_output or (
            audit_json_output.with_suffix(".md") if audit_json_output is not None else None
        )
        if markdown_path is not None:
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(report_to_markdown(report), encoding="utf-8")
            audit_markdown_path = markdown_path

    archive_path: Path | None = None
    if zip_output is not None:
        if workflow_outputs.manifest is None or package_dir is None or skip_package:
            raise ValueError("Zip output requested but package generation was skipped")
        archive_path = _make_archive(zip_output, package_dir)

    return PublishOutputs(
        workflow=workflow_outputs,
        executive_summary=summary_path,
        audit_json=audit_json_path,
        audit_markdown=audit_markdown_path,
        package_archive=archive_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    summary_groups = orchestrator._parse_summary_groups(args.summary_group_by)
    reason_groups = orchestrator._parse_briefing_groups(args.reasonableness_group_by)
    calibration_groups = orchestrator._parse_briefing_groups(args.calibration_group_by)
    calibration_metrics = orchestrator._parse_calibration_metrics(args.calibration_metrics)

    expectations: Sequence[Expectation] | None = None
    if args.audit_config is not None:
        expectations = load_expectations(args.audit_config)

    publish_submission(
        evaluation_path=args.evaluation,
        summary_output=args.summary_output,
        summary_group_cols=summary_groups,
        scenario_output=args.scenario_output,
        scenario_spec=args.scenario_spec,
        reasonableness_output=args.reasonableness_output,
        reasonableness_group_cols=reason_groups,
        macro_config=args.macro_config,
        macro_output=args.macro_output,
        calibration_output=args.calibration_output,
        calibration_group_cols=calibration_groups,
        calibration_metrics=calibration_metrics,
        briefing_output=args.briefing_output,
        coverage_target=args.coverage_target,
        identity_threshold=args.identity_threshold,
        package_dir=None if args.skip_package else args.package_dir,
        skip_package=args.skip_package,
        copy_artifacts=args.copy_artifacts,
        artifact_overrides=args.artifact,
        executive_summary_output=None if args.skip_executive_summary else args.executive_summary_output,
        executive_summary_title=args.executive_summary_title,
        skip_executive_summary=args.skip_executive_summary,
        llm_seed_summary=args.llm_seed_summary,
        loan_pricing_summary=args.loan_pricing_summary,
        credit_metadata=args.credit_metadata,
        risk_summary=args.risk_summary,
        skip_audit=args.skip_audit,
        audit_expectations=expectations,
        audit_json_output=None if args.skip_audit else args.audit_output,
        audit_markdown_output=None if args.skip_audit else args.audit_markdown_output,
        zip_output=args.zip_output,
    )

    logging.info("Publishing workflow completed")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
