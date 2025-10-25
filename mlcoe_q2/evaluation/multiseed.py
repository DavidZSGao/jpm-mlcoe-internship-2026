"""Utilities for aggregating and reporting multi-seed benchmarks."""

from __future__ import annotations

"""Utilities for aggregating and reporting multi-seed benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, MutableMapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MetricSummary:
    """Mean and standard deviation for a metric aggregated across seeds."""

    mean: float | None
    std: float | None

    def as_dict(self) -> MutableMapping[str, float | None]:
        return {"mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class MultiSeedAggregate:
    """Aggregated metrics for a collection of per-seed benchmark runs."""

    num_seeds: int
    metrics: dict[str, dict[str, MetricSummary]]

    def as_dict(self) -> MutableMapping[str, object]:
        return {
            "num_seeds": self.num_seeds,
            "metrics": {
                metric: {
                    method: summary.as_dict()
                    for method, summary in method_summaries.items()
                }
                for metric, method_summaries in self.metrics.items()
            },
        }


def aggregate_metrics(
    per_seed: Sequence[Mapping[str, Mapping[str, Mapping[str, float | None]]]],
    section_key: str,
    metrics: Sequence[str],
) -> MultiSeedAggregate:
    """Aggregate metrics across seeds for the specified section key."""

    num_seeds = len(per_seed)
    aggregated: dict[str, dict[str, MetricSummary]] = {}
    method_order: list[str] = []

    if per_seed:
        first_section = per_seed[0].get(section_key, {})
        method_order = list(first_section.keys())

    for metric in metrics:
        metric_summaries: dict[str, MetricSummary] = {}
        for method in method_order:
            values = []
            for record in per_seed:
                section = record.get(section_key, {})
                method_payload = section.get(method, {})
                value = method_payload.get(metric)
                if value is not None:
                    values.append(float(value))
            if values:
                arr = np.asarray(values, dtype=float)
                summary = MetricSummary(
                    mean=float(arr.mean()),
                    std=float(arr.std(ddof=0)),
                )
            else:
                summary = MetricSummary(mean=None, std=None)
            metric_summaries[method] = summary
        aggregated[metric] = metric_summaries

    return MultiSeedAggregate(num_seeds=num_seeds, metrics=aggregated)


def render_markdown_table(
    *,
    title: str,
    aggregate: MultiSeedAggregate,
    metric_order: Sequence[str],
    column_labels: Mapping[str, str],
    notes: Sequence[str] | None = None,
    missing_value: str = "—",
    formatter_overrides: Mapping[str, Callable[[MetricSummary], str]] | None = None,
) -> str:
    """Render a Markdown table summarising aggregated metrics."""

    formatter_overrides = dict(formatter_overrides or {})

    def default_formatter(summary: MetricSummary) -> str:
        if summary.mean is None:
            return missing_value
        if summary.std is None or summary.std == 0.0:
            return f"{summary.mean:.2f}"
        return f"{summary.mean:.2f} ± {summary.std:.2f}"

    def format_cell(metric: str, summary: MetricSummary) -> str:
        if metric in formatter_overrides:
            return formatter_overrides[metric](summary)
        return default_formatter(summary)

    methods: list[str] = []
    if metric_order:
        first_metric = aggregate.metrics.get(metric_order[0])
    else:
        first_metric = None
    if first_metric:
        methods = list(first_metric.keys())

    header = ["Method"] + [column_labels.get(metric, metric) for metric in metric_order]
    lines = [f"# {title}", "", "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    for method in methods:
        row = [method]
        for metric in metric_order:
            summary = aggregate.metrics.get(metric, {}).get(method, MetricSummary(None, None))
            row.append(format_cell(metric, summary))
        lines.append("| " + " | ".join(row) + " |")

    notes_block = list(notes or [])
    if aggregate.num_seeds:
        notes_block.insert(0, f"- Aggregated across {aggregate.num_seeds} seed(s)")
    if notes_block:
        lines.extend(["", "## Notes"])
        lines.extend(notes_block)

    return "\n".join(lines) + "\n"


__all__ = [
    "MetricSummary",
    "MultiSeedAggregate",
    "aggregate_metrics",
    "render_markdown_table",
]
