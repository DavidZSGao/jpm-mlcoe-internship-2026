"""Audit the Strategic Lending artifact suite for required deliverables."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Expectation:
    """Describe a single artifact that should be present in the repository."""

    name: str
    path: Path
    optional: bool = False
    description: str | None = None


@dataclass(frozen=True)
class AuditResult:
    """Outcome for a single expectation."""

    name: str
    path: str
    relative_path: str
    exists: bool
    optional: bool
    size_bytes: int | None
    modified: str | None
    description: str | None


@dataclass(frozen=True)
class AuditSummary:
    """Aggregate counts for audit results."""

    total: int
    present: int
    missing_required: int
    missing_optional: int


@dataclass(frozen=True)
class AuditReport:
    """Container for audit results and summary statistics."""

    generated_at: str
    summary: AuditSummary
    artifacts: list[AuditResult]


def _default_expectations() -> list[Expectation]:
    """Return the built-in list of required and optional artifacts."""

    return [
        Expectation(
            "Forecaster evaluation", REPO_ROOT / "reports/q1/artifacts/forecaster_evaluation.parquet"
        ),
        Expectation(
            "Forecaster evaluation summary",
            REPO_ROOT / "reports/q1/artifacts/forecaster_evaluation_summary.parquet",
        ),
        Expectation(
            "Scenario reasonableness diagnostics",
            REPO_ROOT / "reports/q1/artifacts/scenario_reasonableness.parquet",
        ),
        Expectation(
            "Macro-conditioned scenarios",
            REPO_ROOT / "reports/q1/artifacts/macro_conditioned_scenarios.parquet",
        ),
        Expectation(
            "Calibration diagnostics", REPO_ROOT / "reports/q1/artifacts/forecaster_calibration.parquet"
        ),
        Expectation(
            "Scenario package", REPO_ROOT / "reports/q1/artifacts/forecaster_scenarios.parquet"
        ),
        Expectation(
            "Strategic Lending briefing", REPO_ROOT / "reports/q1/status/strategic_lending_briefing.md"
        ),
        Expectation(
            "Executive summary", REPO_ROOT / "reports/q1/status/executive_summary.md"
        ),
        Expectation(
            "Risk warning summary", REPO_ROOT / "reports/q1/artifacts/risk_warnings_summary.json"
        ),
        Expectation(
            "Loan pricing summary", REPO_ROOT / "reports/q1/artifacts/loan_pricing_summary.json"
        ),
        Expectation(
            "Credit dataset metadata", REPO_ROOT / "data/credit_ratings/altman_features.json"
        ),
        Expectation(
            "Credit feature parquet",
            REPO_ROOT / "data/credit_ratings/altman_features.parquet",
            description="Altman feature dataset for loan pricing",
        ),
        Expectation(
            "LLM benchmark manifest",
            REPO_ROOT / "reports/q1/artifacts/llm_benchmarks/manifest.json",
            optional=True,
        ),
        Expectation(
            "LLM benchmark summary",
            REPO_ROOT / "reports/q1/artifacts/llm_benchmarks/summary_by_model.parquet",
            optional=True,
        ),
        Expectation(
            "Lending package manifest",
            REPO_ROOT / "reports/q1/deliverables/lending_package_manifest.json",
            optional=True,
        ),
        Expectation(
            "Orchestration log",
            REPO_ROOT / "reports/q1/status/orchestration_report.json",
            optional=True,
            description="JSON status output emitted by orchestrate_lending_workflow",
        ),
    ]


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _format_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return timestamp.isoformat()


def _size_bytes(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size


def audit(expectations: Sequence[Expectation]) -> AuditReport:
    """Evaluate expectations and return an audit report."""

    results: list[AuditResult] = []
    present = 0
    missing_required = 0
    missing_optional = 0

    for expectation in expectations:
        exists = expectation.path.exists()
        if exists:
            present += 1
        elif expectation.optional:
            missing_optional += 1
        else:
            missing_required += 1

        result = AuditResult(
            name=expectation.name,
            path=str(expectation.path),
            relative_path=_relative_path(expectation.path),
            exists=exists,
            optional=expectation.optional,
            size_bytes=_size_bytes(expectation.path),
            modified=_format_timestamp(expectation.path),
            description=expectation.description,
        )
        results.append(result)

    summary = AuditSummary(
        total=len(expectations),
        present=present,
        missing_required=missing_required,
        missing_optional=missing_optional,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    return AuditReport(generated_at=generated_at, summary=summary, artifacts=results)


def report_to_json(report: AuditReport) -> str:
    """Serialize an audit report to a JSON string."""

    return json.dumps(asdict(report), indent=2)


def report_to_markdown(report: AuditReport) -> str:
    """Render an audit report as Markdown."""

    lines = ["# Strategic Lending Artifact Audit", ""]
    lines.append(f"Generated at {report.generated_at}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Total tracked artifacts: {report.summary.total}"
    )
    lines.append(f"- Present: {report.summary.present}")
    lines.append(f"- Missing (required): {report.summary.missing_required}")
    lines.append(f"- Missing (optional): {report.summary.missing_optional}")
    lines.append("")
    lines.append("## Details")
    lines.append("")
    header = "| Artifact | Status | Path | Notes |"
    divider = "| --- | --- | --- | --- |"
    lines.extend([header, divider])

    for result in report.artifacts:
        status = "✅ Present" if result.exists else ("⚠️ Optional" if result.optional else "❌ Missing")
        note = result.description or ""
        if result.exists and result.size_bytes is not None:
            note = f"Size: {result.size_bytes} bytes"
        lines.append(
            f"| {result.name} | {status} | `{result.relative_path}` | {note} |"
        )

    lines.append("")
    return "\n".join(lines)


def load_expectations(path: Path) -> list[Expectation]:
    """Load expectation definitions from a JSON configuration file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expectation config must be a list of objects")

    expectations: list[Expectation] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError("Each expectation must be a JSON object")
        if "name" not in entry or "path" not in entry:
            raise ValueError("Expectation objects require 'name' and 'path' fields")

        raw_path = Path(entry["path"])
        if not raw_path.is_absolute():
            raw_path = REPO_ROOT / raw_path

        expectations.append(
            Expectation(
                name=str(entry["name"]),
                path=raw_path,
                optional=bool(entry.get("optional", False)),
                description=entry.get("description"),
            )
        )

    return expectations


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON file defining artifact expectations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports/q1/status/lending_artifact_audit.json",
        help="Destination for the JSON audit report",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional Markdown destination (defaults to <output>.md)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    expectations = (
        load_expectations(args.config) if args.config else _default_expectations()
    )
    report = audit(expectations)

    json_output = report_to_json(report)
    _ensure_parent(args.output)
    args.output.write_text(json_output, encoding="utf-8")
    logging.info("Wrote audit JSON to %s", args.output)

    markdown_path = args.markdown_output or args.output.with_suffix(".md")
    markdown_output = report_to_markdown(report)
    _ensure_parent(markdown_path)
    markdown_path.write_text(markdown_output, encoding="utf-8")
    logging.info("Wrote audit Markdown to %s", markdown_path)


if __name__ == "__main__":
    main()
