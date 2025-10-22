"""Tests for the workplan progress summariser pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlcoe_q1.pipelines import summarize_workplan_progress


WORKPLAN_SAMPLE = """# Question 1 Execution Plan

## Part 1 — Balance Sheet Modelling
- [x] Deterministic balance-sheet projection spec and implementation.
- [ ] Extend macro driver sampling note.

### Part 2 — LLM-Assisted Analysis
- [x] Baseline adapter ready.
- [ ] Run hosted API benchmarking sweep.

## Bonus Tracks
- [ ] Loan pricing research follow-up.
"""


def test_parse_and_summarize(tmp_path: Path) -> None:
    workplan = tmp_path / "workplan.md"
    workplan.write_text(WORKPLAN_SAMPLE, encoding="utf-8")

    sections, summary = summarize_workplan_progress.summarize_workplan(workplan)

    assert len(sections) == 3
    names = [section.name for section in sections]
    assert names == [
        "Part 1 — Balance Sheet Modelling",
        "Part 1 — Balance Sheet Modelling > Part 2 — LLM-Assisted Analysis",
        "Bonus Tracks",
    ]

    totals = [section.total for section in sections]
    assert totals == [2, 2, 1]

    assert summary["total_tasks"] == 5
    assert summary["completed_tasks"] == 2
    assert summary["remaining_tasks"] == [
        "Extend macro driver sampling note.",
        "Run hosted API benchmarking sweep.",
        "Loan pricing research follow-up.",
    ]


def test_markdown_render(tmp_path: Path) -> None:
    workplan = tmp_path / "workplan.md"
    workplan.write_text(WORKPLAN_SAMPLE, encoding="utf-8")

    output = tmp_path / "progress.md"
    json_output = tmp_path / "progress.json"

    exit_code = summarize_workplan_progress.main(
        [
            "--workplan",
            str(workplan),
            "--output",
            str(output),
            "--json-output",
            str(json_output),
            "--log-level",
            "DEBUG",
        ]
    )

    assert exit_code == 0
    markdown = output.read_text(encoding="utf-8")
    assert "# Question 1 Workplan Progress" in markdown
    assert "Part 1 — Balance Sheet Modelling" in markdown
    assert "Outstanding" in markdown

    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert payload["total_tasks"] == 5
    assert payload["completed_tasks"] == 2


def test_missing_workplan(tmp_path: Path) -> None:
    missing = tmp_path / "absent.md"
    with pytest.raises(FileNotFoundError):
        summarize_workplan_progress.summarize_workplan(missing)
