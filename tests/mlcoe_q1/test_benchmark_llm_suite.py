"""Tests for the LLM benchmarking pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlcoe_q1.pipelines import benchmark_llm_suite, run_llm_adapter


class _StubAdapter:
    def __init__(self, model_name: str | None = None, **_: object) -> None:
        self.name = "stub"
        self.model_id = model_name or "stub-model"
        self._seed: int | None = None

    def set_seed(self, seed: int) -> None:
        self._seed = seed

    def generate(self, prompt: str, *, temperature: float = 0.0, max_new_tokens: int = 256) -> str:
        assert temperature == 0.0
        assert max_new_tokens == 32 or max_new_tokens == 256
        # All prompts share the same target payload for the test.
        return json.dumps({"balance_sheet": {"cash": 1.0, "debt": 0.5}})


def _build_prompt_dataset(path: Path) -> None:
    rows = [
        {
            "ticker": "AAA",
            "context_period": "2022-12-31",
            "target_period": "2023-12-31",
            "prompt": "Project the balance sheet",
            "statements": ["balance_sheet"],
            "ground_truth": {"balance_sheet": {"cash": 1.0, "debt": 0.5}},
        },
        {
            "ticker": "BBB",
            "context_period": "2022-12-31",
            "target_period": "2023-12-31",
            "prompt": "Second prompt",
            "statements": ["balance_sheet"],
            "ground_truth": {"balance_sheet": {"cash": 1.0, "debt": 0.5}},
        },
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_benchmark_llm_suite_end_to_end(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.parquet"
    _build_prompt_dataset(prompts_path)

    monkeypatch.setattr(run_llm_adapter, "create_adapter", lambda *_, **kwargs: _StubAdapter(kwargs.get("model")))

    output_root = tmp_path / "artifacts"
    summary_path = output_root / "summary.json"

    benchmark_llm_suite.main(
        [
            "--prompts",
            str(prompts_path),
            "--output-root",
            str(output_root),
            "--adapter",
            "stub",
            "--models",
            "stub-model",
            "--seeds",
            "1,2",
            "--max-new-tokens",
            "32",
            "--summary-output",
            str(summary_path),
        ]
    )

    responses_dir = output_root / "responses"
    metrics_dir = output_root / "metrics"
    assert responses_dir.exists()
    assert metrics_dir.exists()

    response_files = list(responses_dir.glob("*.parquet"))
    metric_files = list(metrics_dir.glob("*.parquet"))
    assert len(response_files) == 1
    assert len(metric_files) == 1

    metadata_path = response_files[0].with_name(response_files[0].name + ".metadata.json")
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["adapter"] == "stub"
    assert metadata["models"] == ["stub-model"]
    assert metadata["seeds"] == [1, 2]
    assert metadata["records"] == 4
    assert metadata["api_base"] is None

    metrics = pd.read_parquet(metric_files[0])
    assert metrics["mae"].max() == 0.0
    assert metrics["coverage"].min() == 1.0

    assert summary_path.exists()
    summary = sorted(json.loads(summary_path.read_text()), key=lambda item: item.get("seed"))
    assert {entry["seed"] for entry in summary} == {1, 2}
    for record in summary:
        assert record["adapter"] == "stub"
        assert record["model"] == "stub-model"
        assert record["records"] == 2
        assert record["mae_mean"] == 0.0
        assert record["coverage_mean"] == 1.0

    seed_summary_path = output_root / "summary_by_model.parquet"
    assert seed_summary_path.exists()
    seed_summary = pd.read_parquet(seed_summary_path)
    assert list(seed_summary["seed_count"]) == [2]
    assert list(seed_summary["mae_std"]) == [0.0]
    assert list(seed_summary["coverage_std"]) == [0.0]

    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["summary"] == str(summary_path)
    assert manifest["seed_summary"] == str(seed_summary_path)
    run_entry = manifest["runs"][0]
    assert run_entry["records"] == 4
    assert run_entry["metadata_path"] == str(metadata_path)
    assert run_entry["metadata_models"] == ["stub-model"]
    assert run_entry["metadata_seeds"] == [1, 2]
    assert run_entry["temperature"] == 0.0
    assert run_entry["max_new_tokens"] == 32
    assert run_entry["api_base"] is None
    assert run_entry["api_key_env"] is None


def test_benchmark_llm_suite_supports_config(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.parquet"
    _build_prompt_dataset(prompts_path)

    monkeypatch.setattr(run_llm_adapter, "create_adapter", lambda *_, **kwargs: _StubAdapter(kwargs.get("model")))

    output_root = tmp_path / "suite"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "prompts": str(prompts_path),
                "output_root": str(output_root),
                "adapter": "stub",
                "models": "stub-model",
                "seeds": "1",
            }
        ),
        encoding="utf-8",
    )

    benchmark_llm_suite.main(["--config", str(config_path)])

    assert (output_root / "responses").exists()
    assert (output_root / "metrics").exists()
