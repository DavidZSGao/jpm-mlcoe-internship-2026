"""Assemble Strategic Lending deliverables into a reproducible package."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


DEFAULT_ARTIFACTS = {
    "strategic_briefing": REPO_ROOT
    / "reports/q1/status/strategic_lending_briefing.md",
    "scenario_table": REPO_ROOT
    / "reports/q1/artifacts/forecaster_scenarios.parquet",
    "scenario_reasonableness": REPO_ROOT
    / "reports/q1/artifacts/scenario_reasonableness.parquet",
    "macro_overlay": REPO_ROOT
    / "reports/q1/artifacts/macro_conditioned_scenarios.parquet",
    "macro_config": REPO_ROOT
    / "reports/q1/artifacts/macro_scenarios_example.json",
    "calibration": REPO_ROOT
    / "reports/q1/artifacts/forecaster_calibration.parquet",
    "evaluation_summary": REPO_ROOT
    / "reports/q1/artifacts/forecaster_evaluation_summary.parquet",
    "raw_evaluation": REPO_ROOT
    / "reports/q1/artifacts/forecaster_evaluation.parquet",
    "credit_dataset": REPO_ROOT / "data/credit_ratings/altman_features.parquet",
    "credit_case_study": REPO_ROOT
    / "reports/q1/artifacts/evergrande_credit_rating.json",
}


@dataclass(frozen=True)
class ArtifactRecord:
    """Metadata describing a single artifact path."""

    label: str
    path: Path
    exists: bool
    is_file: bool
    size_bytes: int | None
    sha256: str | None
    relative_path: str
    package_path: str | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build CLI arguments for the lending-package compiler."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Optional artifact override in LABEL=PATH form. "
            "Multiple flags may be supplied to add or replace defaults."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/q1/deliverables",
        help="Directory where the manifest, README, and copied artifacts are stored",
    )
    parser.add_argument(
        "--copy-artifacts",
        action="store_true",
        help="Copy resolved artifacts into the output directory for offline bundling",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level to use for status messages",
    )
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "artifact": list,
            "output_dir": Path,
        },
    )


def _parse_artifact_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            "Artifact overrides must be provided as LABEL=PATH (missing '=')."
        )
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("Artifact label cannot be empty")
    candidate = Path(path.strip()).expanduser()
    return label, candidate


def _load_artifact_map(overrides: Iterable[str]) -> dict[str, Path]:
    mapping = {label: path for label, path in DEFAULT_ARTIFACTS.items()}
    for spec in overrides:
        label, path = _parse_artifact_spec(spec)
        mapping[label] = path
    return mapping


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:  # pragma: no cover - handles external paths
        return str(path)


def collect_artifacts(
    artifact_map: Mapping[str, Path],
    repo_root: Path | None = None,
) -> list[ArtifactRecord]:
    """Gather file metadata for the configured artifact set."""

    repo_root = repo_root or REPO_ROOT
    records: list[ArtifactRecord] = []
    for label in sorted(artifact_map):
        path = artifact_map[label]
        resolved = path.expanduser()
        if not resolved.is_absolute():
            resolved = (repo_root / resolved).resolve()
        exists = resolved.exists()
        is_file = resolved.is_file() if exists else False
        size = resolved.stat().st_size if exists and is_file else None
        record = ArtifactRecord(
            label=label,
            path=resolved,
            exists=exists,
            is_file=is_file,
            size_bytes=size,
            sha256=_sha256(resolved),
            relative_path=_relative_to_repo(resolved, repo_root),
            package_path=None,
        )
        records.append(record)
    return records


def _sanitise_label(label: str) -> str:
    safe = []
    for ch in label.strip():
        if not ch.strip():
            continue
        if ch.isalnum():
            safe.append(ch.lower())
        elif ch in {"-", "_"}:
            safe.append(ch)
        else:
            safe.append("-")
    token = "".join(safe).strip("-")
    return token or "artifact"


def copy_artifacts(records: list[ArtifactRecord], output_dir: Path) -> list[ArtifactRecord]:
    """Copy existing artifact files into the output directory."""

    if not records:
        return records

    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    updated: list[ArtifactRecord] = []
    for record in records:
        if not record.exists or not record.is_file:
            updated.append(record)
            continue
        safe_label = _sanitise_label(record.label)
        suffix = record.path.suffix
        candidate = artifact_dir / f"{safe_label}{suffix}"
        if candidate.exists():
            candidate = artifact_dir / f"{safe_label}_{record.path.name}"
        shutil.copy2(record.path, candidate)
        updated.append(
            ArtifactRecord(
                label=record.label,
                path=record.path,
                exists=record.exists,
                is_file=record.is_file,
                size_bytes=record.size_bytes,
                sha256=record.sha256,
                relative_path=record.relative_path,
                package_path=str(candidate.relative_to(output_dir)),
            )
        )
    return updated


def write_manifest(records: Sequence[ArtifactRecord], output_dir: Path) -> Path:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {
                "label": record.label,
                "path": record.relative_path,
                "exists": record.exists,
                "is_file": record.is_file,
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "package_path": record.package_path,
            }
            for record in records
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "lending_package_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2))
    return destination


def write_readme(records: Sequence[ArtifactRecord], output_dir: Path) -> Path:
    lines = ["# Strategic Lending Deliverable Package", ""]
    lines.append(
        "This directory captures the artifacts required to brief the Strategic "
        "Lending Division on current forecast performance, scenario coverage, "
        "and supporting credit analytics."
    )
    lines.extend(["", "| Label | Exists | Path | Size (bytes) | Packaged Path |", "| --- | --- | --- | --- | --- |"])
    for record in records:
        package_path = record.package_path or "—"
        size = record.size_bytes if record.size_bytes is not None else "—"
        lines.append(
            f"| {record.label} | {'✅' if record.exists else '❌'} | "
            f"`{record.relative_path}` | {size} | {package_path} |"
        )
    destination = output_dir / "README.md"
    destination.write_text("\n".join(lines))
    return destination


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")

    artifact_map = _load_artifact_map(args.artifact)
    records = collect_artifacts(artifact_map)
    if args.copy_artifacts:
        records = copy_artifacts(records, args.output_dir)
    manifest_path = write_manifest(records, args.output_dir)
    readme_path = write_readme(records, args.output_dir)
    logging.info("Manifest written to %s", manifest_path)
    logging.info("README written to %s", readme_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

