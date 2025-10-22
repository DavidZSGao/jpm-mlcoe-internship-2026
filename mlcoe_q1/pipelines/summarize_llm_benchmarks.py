"""Summarise LLM benchmark manifests into ranked tables and Markdown briefs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--manifest", type=Path, required=True, help="Benchmark manifest path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Ranked summary output path (defaults to manifest directory)",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Markdown briefing output path (defaults to manifest directory)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top-performing models to highlight in Markdown output",
    )
    parser.add_argument("--log-level", default="INFO")
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={"manifest": Path, "output": Path, "markdown_output": Path},
    )


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path, orient="records")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rank_models(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    if "mae_mean" in ranked.columns:
        ranked["mae_rank"] = ranked["mae_mean"].rank(method="min")
    if "coverage_error" not in ranked.columns and "coverage_mean" in ranked.columns:
        ranked["coverage_error"] = (ranked["coverage_mean"] - 1.0).abs()
    if "coverage_error" in ranked.columns:
        ranked["coverage_rank"] = ranked["coverage_error"].rank(method="min")
    if "mape_mean" in ranked.columns:
        ranked["mape_rank"] = ranked["mape_mean"].rank(method="min")
    sort_columns: list[str] = []
    for column in ("mae_rank", "coverage_rank", "mape_rank"):
        if column in ranked.columns:
            sort_columns.append(column)
    if sort_columns:
        ranked = ranked.sort_values(sort_columns, na_position="last")
    ranked = ranked.reset_index(drop=True)
    return ranked


def _format_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = " | ".join(columns)
    separator = " | ".join(["---"] * len(columns))
    lines = [f"| {header} |", f"| {separator} |"]
    for _, row in df.iterrows():
        cells = [row[col] for col in columns]
        formatted = []
        for cell in cells:
            if isinstance(cell, float):
                formatted.append(f"{cell:.4f}")
            else:
                formatted.append(str(cell))
        lines.append(f"| {' | '.join(formatted)} |")
    return "\n".join(lines)


def _build_markdown(
    ranked: pd.DataFrame,
    summary: pd.DataFrame,
    manifest: dict[str, object],
    top_n: int,
) -> str:
    total_models = ranked.shape[0]
    adapters = ranked["adapter"].nunique() if "adapter" in ranked.columns else 0
    total_records = sum(run.get("records", 0) for run in manifest.get("runs", []))
    seeds = ranked.get("seed_count")
    seed_summary = (
        int(seeds.max()) if seeds is not None and not pd.isna(seeds.max()) else None
    )
    lines = ["# LLM Benchmark Summary", ""]
    lines.append(f"*Adapters evaluated:* {adapters}")
    lines.append(f"*Model configurations:* {total_models}")
    lines.append(f"*Prompt evaluations:* {total_records}")
    if seed_summary:
        lines.append(f"*Seeds per configuration (max):* {seed_summary}")
    lines.append("")

    highlight = ranked.head(top_n)
    columns = ["adapter", "model", "mae_mean", "mape_mean", "coverage_mean", "records"]
    columns = [c for c in columns if c in highlight.columns]
    if not highlight.empty and columns:
        lines.append("## Top Configurations by MAE")
        lines.append("")
        lines.append(_format_markdown_table(highlight, columns))
        lines.append("")

    if "seed" in summary.columns:
        variability = (
            summary.groupby(["adapter", "model"])["mae"].std(ddof=0).reset_index(name="mae_std")
        )
        variability = variability.sort_values("mae_std")
        if not variability.empty:
            lines.append("## Seed Variability (Lower is Better)")
            lines.append("")
            columns = ["adapter", "model", "mae_std"]
            variability = variability.head(top_n)
            variability["mae_std"] = variability["mae_std"].fillna(0.0)
            lines.append(_format_markdown_table(variability, columns))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    summary_path = Path(manifest_data["summary"])
    seed_summary_path = Path(manifest_data["seed_summary"])

    summary = _load_table(summary_path)
    ranked = _load_table(seed_summary_path)
    ranked = _rank_models(ranked)

    if args.output is None:
        args.output = summary_path.parent / "benchmark_ranked.parquet"
    if args.markdown_output is None:
        args.markdown_output = summary_path.parent / "benchmark_summary.md"

    _ensure_parent(args.output)
    ranked.to_parquet(args.output, index=False)

    markdown = _build_markdown(ranked, summary, manifest_data, args.top_n)
    _ensure_parent(args.markdown_output)
    args.markdown_output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

