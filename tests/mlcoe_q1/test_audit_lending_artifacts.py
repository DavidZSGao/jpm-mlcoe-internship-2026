from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from mlcoe_q1.pipelines.audit_lending_artifacts import (
    Expectation,
    audit,
    load_expectations,
    report_to_json,
    report_to_markdown,
)


def test_audit_counts_and_metadata(tmp_path: Path) -> None:
    required_file = tmp_path / "required.parquet"
    required_file.write_bytes(b"data")

    optional_file = tmp_path / "optional.json"

    expectations = [
        Expectation(name="Required", path=required_file),
        Expectation(name="Optional", path=optional_file, optional=True),
    ]

    report = audit(expectations)

    assert report.summary.total == 2
    assert report.summary.present == 1
    assert report.summary.missing_required == 0
    assert report.summary.missing_optional == 1

    result_map = {item.name: item for item in report.artifacts}
    assert result_map["Required"].exists is True
    assert result_map["Required"].size_bytes == 4
    assert result_map["Optional"].exists is False
    assert result_map["Optional"].optional is True

    # Ensure timestamps are ISO formatted when the file exists.
    timestamp = datetime.fromisoformat(result_map["Required"].modified)
    assert timestamp.tzinfo is not None


def test_report_serialisation(tmp_path: Path) -> None:
    expectation = Expectation(name="File", path=tmp_path / "file.txt")
    report = audit([expectation])

    json_payload = json.loads(report_to_json(report))
    assert json_payload["summary"]["total"] == 1

    markdown_output = report_to_markdown(report)
    assert "Strategic Lending Artifact Audit" in markdown_output
    assert "File" in markdown_output
    assert "❌ Missing" in markdown_output


def test_load_expectations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "name": "Relative",
                    "path": "reports/q1/artifacts/test.parquet",
                },
                {
                    "name": "Absolute",
                    "path": str(tmp_path / "absolute.json"),
                    "optional": True,
                    "description": "sample",
                },
            ]
        ),
        encoding="utf-8",
    )

    expectations = load_expectations(config_path)
    assert len(expectations) == 2
    assert expectations[0].path.is_absolute()
    assert expectations[1].optional is True
    assert expectations[1].description == "sample"
