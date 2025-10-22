"""Tests for the Strategic Lending deliverable packager."""

from __future__ import annotations

import hashlib
from pathlib import Path

from mlcoe_q1.pipelines.compile_lending_package import (
    ArtifactRecord,
    collect_artifacts,
    copy_artifacts,
    write_manifest,
    write_readme,
)


def test_collect_artifacts_records(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("hello world")

    records = collect_artifacts({"briefing": target}, repo_root=tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record.label == "briefing"
    assert record.exists is True
    assert record.is_file is True
    assert record.size_bytes == target.stat().st_size
    assert record.sha256 == hashlib.sha256(b"hello world").hexdigest()
    assert record.relative_path == "example.txt"


def test_collect_artifacts_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    records = collect_artifacts({"scenario": missing}, repo_root=tmp_path)
    record = records[0]
    assert record.exists is False
    assert record.size_bytes is None
    assert record.sha256 is None


def test_copy_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}")  # simple placeholder
    record = ArtifactRecord(
        label="macro_config",
        path=source,
        exists=True,
        is_file=True,
        size_bytes=source.stat().st_size,
        sha256="dummy",
        relative_path="source.json",
        package_path=None,
    )

    output_dir = tmp_path / "out"
    updated_records = copy_artifacts([record], output_dir)
    packaged = output_dir / "artifacts" / "macro_config.json"
    assert packaged.exists()
    assert updated_records[0].package_path == "artifacts/macro_config.json"


def test_write_manifest_and_readme(tmp_path: Path) -> None:
    record = ArtifactRecord(
        label="briefing",
        path=tmp_path / "briefing.md",
        exists=False,
        is_file=False,
        size_bytes=None,
        sha256=None,
        relative_path="reports/q1/status/strategic_lending_briefing.md",
        package_path=None,
    )
    manifest = write_manifest([record], tmp_path)
    readme = write_readme([record], tmp_path)

    assert manifest.exists()
    assert readme.exists()
    content = readme.read_text()
    assert "Strategic Lending Deliverable Package" in content
    assert "briefing" in content

