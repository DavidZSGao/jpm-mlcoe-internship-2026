"""Summarise completion status for the Question 1 execution workplan."""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


@dataclass
class Task:
    """Represents a single checkbox entry in the workplan."""

    description: str
    completed: bool


@dataclass
class SectionProgress:
    """Aggregated completion information for a logical workplan section."""

    name: str
    tasks: list[Task]

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def completed(self) -> int:
        return sum(1 for task in self.tasks if task.completed)

    @property
    def remaining(self) -> list[str]:
        return [task.description for task in self.tasks if not task.completed]

    @property
    def completion_rate(self) -> float | None:
        if not self.tasks:
            return None
        return self.completed / self.total


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--workplan",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports/q1/q1_workplan.md",
        help="Path to the Question 1 workplan markdown file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/status/workplan_progress.md",
        help="Optional Markdown output path for the summarised progress report",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON output path for machine-readable progress metadata",
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
            "workplan": Path,
            "output": Path,
            "json_output": Path,
        },
    )


def _normalise_section_name(stack: Iterable[tuple[int, str]]) -> str:
    titles = [title for level, title in stack if level >= 2]
    return " > ".join(titles) if titles else "General"


def _parse_workplan_markdown(workplan: Path) -> list[SectionProgress]:
    if not workplan.exists():
        raise FileNotFoundError(f"Workplan file not found: {workplan}")

    sections: "OrderedDict[str, list[Task]]" = OrderedDict()
    section_order: list[str] = []
    heading_stack: list[tuple[int, str]] = []

    for raw_line in workplan.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            hash_count = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            while heading_stack and heading_stack[-1][0] >= hash_count:
                heading_stack.pop()
            heading_stack.append((hash_count, title))
            continue

        if "[" not in stripped:
            continue

        if "- [" not in stripped and "* [" not in stripped:
            continue

        marker_index = stripped.find("[")
        status_token = stripped[marker_index + 1 : marker_index + 2].lower()
        if status_token not in {"x", " "}:
            continue

        description = stripped[marker_index + 3 :].strip()
        section_name = _normalise_section_name(heading_stack)
        if section_name not in sections:
            sections[section_name] = []
            section_order.append(section_name)
        sections[section_name].append(
            Task(description=description, completed=status_token == "x")
        )

    return [SectionProgress(name=section, tasks=sections[section]) for section in section_order]


def _build_summary(sections: list[SectionProgress]) -> dict[str, object]:
    total_tasks = sum(section.total for section in sections)
    completed_tasks = sum(section.completed for section in sections)
    summary: dict[str, object] = {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "completion_rate": (completed_tasks / total_tasks) if total_tasks else None,
        "sections": [],
    }
    for section in sections:
        summary["sections"].append(
            {
                "name": section.name,
                "total_tasks": section.total,
                "completed_tasks": section.completed,
                "completion_rate": section.completion_rate,
                "remaining_tasks": section.remaining,
            }
        )
    summary["remaining_tasks"] = [
        task
        for section in summary["sections"]
        for task in section["remaining_tasks"]
    ]
    return summary


def _format_percentage(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _render_markdown(summary: dict[str, object]) -> str:
    lines = ["# Question 1 Workplan Progress", ""]
    total = summary.get("total_tasks", 0)
    completed = summary.get("completed_tasks", 0)
    completion_rate = _format_percentage(summary.get("completion_rate"))
    if not total:
        lines.append("No checkbox tasks were found in the workplan.")
        return "\n".join(lines)

    lines.append(f"- Completed: {completed}/{total} ({completion_rate})")
    remaining_tasks = summary.get("remaining_tasks", [])
    lines.append(f"- Remaining tasks: {len(remaining_tasks)}")
    lines.append("")

    for section in summary.get("sections", []):
        section_name = section["name"]
        lines.append(f"## {section_name}")
        lines.append("")
        sec_completed = section["completed_tasks"]
        sec_total = section["total_tasks"]
        sec_rate = _format_percentage(section["completion_rate"])
        lines.append(f"- Completed: {sec_completed}/{sec_total} ({sec_rate})")
        remaining = section["remaining_tasks"]
        if remaining:
            lines.append("- Outstanding:")
            for task in remaining:
                lines.append(f"  - {task}")
        else:
            lines.append("- Outstanding: None")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def summarize_workplan(workplan: Path) -> tuple[list[SectionProgress], dict[str, object]]:
    sections = _parse_workplan_markdown(workplan)
    summary = _build_summary(sections)
    return sections, summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    logging.info("Loading workplan from %s", args.workplan)
    sections, summary = summarize_workplan(args.workplan)

    markdown = _render_markdown(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        logging.info("Wrote Markdown summary to %s", args.output)
    else:
        print(markdown)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logging.info("Wrote JSON summary to %s", args.json_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

