"""Run the end-to-end Strategic Lending workflow from a single CLI."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from mlcoe_q1.pipelines.analyze_forecaster_calibration import (
    _parse_tokens as _parse_calibration_metrics,
    compute_calibration,
)
from mlcoe_q1.pipelines.assess_scenario_reasonableness import (
    compute_scenario_statistics,
)
from mlcoe_q1.pipelines.compile_lending_package import (
    DEFAULT_ARTIFACTS,
    collect_artifacts,
    copy_artifacts,
    write_manifest,
    write_readme,
)
from mlcoe_q1.pipelines.generate_lending_briefing import (
    build_briefing,
    _parse_group_columns as _parse_briefing_groups,
)
from mlcoe_q1.pipelines.package_scenarios import (
    _parse_scenarios as _parse_scenario_spec,
    build_scenarios,
)
from mlcoe_q1.pipelines.simulate_macro_conditions import (
    apply_macro_scenarios,
    load_macro_scenarios,
)
from mlcoe_q1.pipelines.summarize_forecaster_evaluation import (
    _parse_group_columns as _parse_summary_groups,
    summarize,
)
from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkflowOutputs:
    """Paths emitted by the orchestration workflow."""

    summary: Path
    scenarios: Path
    macro: Path
    reasonableness: Path
    calibration: Path
    briefing: Path
    manifest: Path | None
    readme: Path | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Configure CLI arguments for the workflow orchestrator."""

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
        default=REPO_ROOT
        / "reports/q1/artifacts/forecaster_evaluation_summary.parquet",
        help="Destination path for the aggregated evaluation summary",
    )
    parser.add_argument(
        "--summary-group-by",
        type=str,
        default="ticker,mode",
        help="Comma separated columns for aggregating the evaluation summary",
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
        help="Comma separated scenario_name:quantile pairs used when packaging",
    )
    parser.add_argument(
        "--reasonableness-output",
        type=Path,
        default=REPO_ROOT
        / "reports/q1/artifacts/scenario_reasonableness.parquet",
        help="Destination parquet for scenario reasonableness diagnostics",
    )
    parser.add_argument(
        "--reasonableness-group-by",
        type=str,
        default="scenario",
        help="Grouping columns for scenario reasonableness aggregation",
    )
    parser.add_argument(
        "--macro-config",
        type=Path,
        default=None,
        help="Optional JSON file specifying macro scenarios to apply",
    )
    parser.add_argument(
        "--macro-output",
        type=Path,
        default=REPO_ROOT
        / "reports/q1/artifacts/macro_conditioned_scenarios.parquet",
        help="Destination parquet for macro-conditioned scenarios",
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=REPO_ROOT
        / "reports/q1/artifacts/forecaster_calibration.parquet",
        help="Destination parquet containing calibration diagnostics",
    )
    parser.add_argument(
        "--calibration-group-by",
        type=str,
        default="ticker,mode",
        help="Grouping columns for calibration metrics",
    )
    parser.add_argument(
        "--calibration-metrics",
        type=str,
        default="assets,equity,net_income",
        help="Comma separated list of metrics to analyse in calibration",
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
        help="Target coverage level flagged inside the briefing",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=5e8,
        help="Absolute accounting identity threshold for briefing warnings",
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
        help="Skip manifest/README generation and artifact bundling",
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="Copy resolved artifacts into the package directory",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional artifact overrides included in the manifest",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level to use for status messages",
    )
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={"macro_config": Path},
    )


def _parse_artifact_override(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            "Artifact overrides must be provided as LABEL=PATH (missing '=')."
        )
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("Artifact label cannot be empty")
    path = Path(raw_path.strip()).expanduser()
    return label, path


def _build_artifact_map(
    overrides: Sequence[str],
    evaluation: Path,
    summary: Path,
    scenarios: Path,
    macro: Path,
    reasonableness: Path,
    calibration: Path,
    briefing: Path,
) -> Mapping[str, Path]:
    mapping = {label: path for label, path in DEFAULT_ARTIFACTS.items()}
    mapping.update(
        {
            "raw_evaluation": evaluation,
            "evaluation_summary": summary,
            "scenario_table": scenarios,
            "macro_overlay": macro,
            "scenario_reasonableness": reasonableness,
            "calibration": calibration,
            "strategic_briefing": briefing,
        }
    )
    for spec in overrides:
        label, path = _parse_artifact_override(spec)
        mapping[label] = path
    return mapping


def run_workflow(
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
    copy_package_artifacts: bool,
    artifact_overrides: Sequence[str],
) -> WorkflowOutputs:
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation parquet not found: {evaluation_path}")

    logging.info("Loading evaluation data from %s", evaluation_path)
    evaluation_df = pd.read_parquet(evaluation_path)
    if evaluation_df.empty:
        raise ValueError("Evaluation dataframe is empty; run evaluate_forecaster first")

    logging.info("Summarising evaluation metrics")
    summary_df = summarize(evaluation_df, summary_group_cols)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(summary_output, index=False)

    logging.info("Packaging scenarios using spec %s", scenario_spec)
    scenario_map = _parse_scenario_spec(scenario_spec)
    scenarios_df = build_scenarios(evaluation_df, scenario_map)
    scenario_output.parent.mkdir(parents=True, exist_ok=True)
    scenarios_df.to_parquet(scenario_output, index=False)

    logging.info("Computing scenario reasonableness diagnostics")
    reason_df = compute_scenario_statistics(scenarios_df, reasonableness_group_cols)
    reasonableness_output.parent.mkdir(parents=True, exist_ok=True)
    reason_df.to_parquet(reasonableness_output, index=False)

    logging.info("Applying macro conditioning to scenarios")
    macro_scenarios = load_macro_scenarios(macro_config)
    macro_df = apply_macro_scenarios(scenarios_df, macro_scenarios)
    macro_output.parent.mkdir(parents=True, exist_ok=True)
    macro_df.to_parquet(macro_output, index=False)

    logging.info("Evaluating calibration metrics")
    calibration_df = compute_calibration(
        evaluation_df, calibration_group_cols, calibration_metrics
    )
    calibration_output.parent.mkdir(parents=True, exist_ok=True)
    calibration_df.to_parquet(calibration_output, index=False)

    logging.info("Generating Strategic Lending briefing")
    briefing_text = build_briefing(
        summary_df, summary_group_cols, coverage_target, identity_threshold
    )
    briefing_output.parent.mkdir(parents=True, exist_ok=True)
    briefing_output.write_text(briefing_text)

    manifest_path: Path | None = None
    readme_path: Path | None = None
    if not skip_package and package_dir is not None:
        logging.info("Compiling deliverable package in %s", package_dir)
        artifact_map = _build_artifact_map(
            artifact_overrides,
            evaluation_path,
            summary_output,
            scenario_output,
            macro_output,
            reasonableness_output,
            calibration_output,
            briefing_output,
        )
        records = collect_artifacts(artifact_map)
        if copy_package_artifacts:
            records = copy_artifacts(records, package_dir)
        manifest_path = write_manifest(records, package_dir)
        readme_path = write_readme(records, package_dir)

    return WorkflowOutputs(
        summary=summary_output,
        scenarios=scenario_output,
        macro=macro_output,
        reasonableness=reasonableness_output,
        calibration=calibration_output,
        briefing=briefing_output,
        manifest=manifest_path,
        readme=readme_path,
    )


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s %(message)s",
    )

    summary_groups = _parse_summary_groups(args.summary_group_by)
    reason_groups = _parse_briefing_groups(args.reasonableness_group_by)
    calibration_groups = _parse_briefing_groups(args.calibration_group_by)
    calibration_metrics = _parse_calibration_metrics(args.calibration_metrics)

    run_workflow(
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
        copy_package_artifacts=args.copy_artifacts,
        artifact_overrides=args.artifact,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

