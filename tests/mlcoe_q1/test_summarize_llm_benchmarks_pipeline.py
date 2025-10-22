from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlcoe_q1.pipelines import summarize_llm_benchmarks


def _write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def test_summarize_llm_benchmarks_creates_ranked_outputs(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "bench"
    manifest_dir.mkdir(parents=True)

    summary_df = pd.DataFrame(
        {
            "adapter": ["hf", "hf", "hf"],
            "model": ["m1", "m1", "m2"],
            "seed": [0, 1, 0],
            "mae": [0.3, 0.4, 0.6],
        }
    )
    seed_summary_df = pd.DataFrame(
        {
            "adapter": ["hf", "hf"],
            "model": ["m1", "m2"],
            "mae_mean": [0.35, 0.6],
            "mape_mean": [0.1, 0.3],
            "coverage_mean": [0.95, 0.8],
            "records": [2, 1],
            "seed_count": [2, 1],
        }
    )

    summary_path = _write_table(summary_df, manifest_dir / "summary.parquet")
    seed_summary_path = _write_table(seed_summary_df, manifest_dir / "summary_by_model.parquet")

    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {"adapter": "hf", "model": "m1", "records": 2},
                    {"adapter": "hf", "model": "m2", "records": 1},
                ],
                "summary": str(summary_path),
                "seed_summary": str(seed_summary_path),
            }
        ),
        encoding="utf-8",
    )

    output_path = manifest_dir / "ranked.parquet"
    markdown_path = manifest_dir / "summary.md"

    summarize_llm_benchmarks.main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    ranked = pd.read_parquet(output_path)
    assert "mae_rank" in ranked.columns
    assert ranked.iloc[0]["model"] == "m1"

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Top Configurations" in markdown
    assert "m1" in markdown
    assert "model configurations" in markdown.lower()

