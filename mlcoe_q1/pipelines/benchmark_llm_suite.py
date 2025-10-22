"""Run a configured suite of LLM adapters and aggregate evaluation metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from mlcoe_q1.pipelines import evaluate_llm_responses, run_llm_adapter
from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmarking CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--prompts", type=Path, required=True, help="Prompt dataset path")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to hold responses, metrics, and summary artifacts",
    )
    parser.add_argument("--adapter", default="flan-t5", help="Adapter registry name")
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model identifiers (falls back to adapter default when omitted)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated random seeds to sweep (single deterministic pass when omitted)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of prompts to sample")
    parser.add_argument(
        "--api-base",
        type=str,
        default="",
        help="Optional base URL override for hosted adapters (e.g. OpenAI-compatible endpoints)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable supplying API keys for hosted adapters",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Request timeout (seconds) for hosted adapters",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional path for per-seed benchmark metrics (defaults to output-root/summary.parquet)",
    )
    parser.add_argument(
        "--seed-summary-output",
        type=Path,
        help="Optional path for seed-aggregated metrics (defaults to output-root/summary_by_model.parquet)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip inference if the target response file already exists",
    )
    parser.add_argument("--log-level", default="INFO")
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "prompts": Path,
            "output_root": Path,
            "summary_output": Path,
            "seed_summary_output": Path,
        },
    )


def _normalise_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _response_filename(adapter: str, model: str | None, seeds: Iterable[int | None]) -> str:
    seed_tokens = sorted({"seed" if seed is None else f"seed{seed}" for seed in seeds})
    seed_suffix = "-".join(seed_tokens) if seed_tokens else "seed"
    model_token = model or "default"
    return f"{adapter}-{model_token}-{seed_suffix}.parquet"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _metadata_path_for(output: Path) -> Path:
    return output.with_name(output.name + ".metadata.json")


def _parse_seed_list(raw: str) -> list[int | None]:
    values: list[int | None] = []
    tokens = _normalise_list(raw)
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError as exc:  # pragma: no cover - defensive validation
            raise ValueError(f"Invalid seed value: {token}") from exc
    if not values:
        values = [None]
    return values


def _aggregate_metrics(df: pd.DataFrame, include_seed: bool) -> pd.DataFrame:
    """Aggregate evaluation metrics with or without the seed dimension."""

    group_cols = ["adapter"]
    if "model" in df.columns:
        group_cols.append("model")
    if include_seed and "seed" in df.columns:
        group_cols.append("seed")

    aggregations: dict[str, object] = {
        "mae": ["mean", "median"],
        "mape": ["mean", "median"],
        "coverage": ["mean", "median"],
        "missing_items": "mean",
        "extra_items": "mean",
        "invalid_items": "mean",
        "matched_items": "sum",
        "total_items": "sum",
    }
    if not include_seed:
        for metric in ("mae", "mape", "coverage"):
            aggregations[metric] = ["mean", "median", "std"]

    grouped = df.groupby(group_cols, dropna=False).agg(aggregations)
    grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    grouped["records"] = df.groupby(group_cols, dropna=False)["mae"].size().values
    if not include_seed and "seed" in df.columns:
        grouped["seed_count"] = (
            df.groupby(group_cols, dropna=False)["seed"].nunique().values
        )
    grouped["coverage_error"] = (grouped["coverage_mean"] - 1.0).abs()
    return grouped


def _save_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".json":
        _ensure_parent(path)
        df.to_json(path, orient="records", indent=2)
    elif path.suffix.lower() == ".csv":
        _ensure_parent(path)
        df.to_csv(path, index=False)
    else:
        _ensure_parent(path)
        df.to_parquet(path, index=False)


def _run_inference(
    args: argparse.Namespace,
    model: str | None,
    seeds: list[int | None],
    output: Path,
    metadata_path: Path,
) -> None:
    cli_args = [
        "--prompts",
        str(args.prompts),
        "--output",
        str(output),
        "--adapter",
        args.adapter,
        "--temperature",
        str(args.temperature),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.limit:
        cli_args.extend(["--limit", str(args.limit)])
    if model:
        cli_args.extend(["--model", model])
    if args.seeds:
        cli_args.extend(["--seeds", args.seeds])
    if args.api_base:
        cli_args.extend(["--api-base", args.api_base])
    if args.api_key_env:
        cli_args.extend(["--api-key-env", args.api_key_env])
    if args.request_timeout:
        cli_args.extend(["--request-timeout", str(args.request_timeout)])
    cli_args.extend(["--metadata-output", str(metadata_path)])

    logging.info("Running adapter %s model %s -> %s", args.adapter, model or "default", output)
    run_llm_adapter.main(cli_args)


def _ensure_metadata(
    args: argparse.Namespace,
    model: str | None,
    responses_path: Path,
    metadata_path: Path,
    responses: pd.DataFrame,
) -> dict[str, object]:
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("Metadata at %s is not valid JSON; regenerating", metadata_path)

    metadata = run_llm_adapter.build_metadata(
        adapter=args.adapter,
        prompts_path=args.prompts,
        output_path=responses_path,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        api_base=args.api_base or None,
        api_key_env=args.api_key_env,
        request_timeout=args.request_timeout,
        responses=responses,
        model_argument=model,
        seed_argument=args.seeds or None,
    )
    _ensure_parent(metadata_path)
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def _load_prompts(path: Path, limit: int) -> pd.DataFrame:
    prompts = pd.read_parquet(path)
    if limit > 0:
        prompts = prompts.head(limit)
    return prompts


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    prompts = _load_prompts(args.prompts, args.limit)

    models = _normalise_list(args.models)
    if not models:
        models = [None]
    seeds = _parse_seed_list(args.seeds)

    responses_dir = args.output_root / "responses"
    metrics_dir = args.output_root / "metrics"

    all_metrics: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []

    for model in models:
        response_name = _response_filename(args.adapter, model, seeds)
        responses_path = responses_dir / response_name
        metrics_path = metrics_dir / response_name
        metadata_path = _metadata_path_for(responses_path)

        if args.skip_existing and responses_path.exists():
            logging.info("Skipping existing responses at %s", responses_path)
        else:
            _ensure_parent(responses_path)
            _run_inference(args, model, seeds, responses_path, metadata_path)

        responses = pd.read_parquet(responses_path)
        metrics = evaluate_llm_responses.evaluate_responses(
            prompts,
            responses,
            response_column="response",
            model_column="model",
        )
        _ensure_parent(metrics_path)
        metrics.to_parquet(metrics_path, index=False)
        all_metrics.append(metrics)
        metadata = _ensure_metadata(args, model, responses_path, metadata_path, responses)
        manifests.append(
            {
                "adapter": args.adapter,
                "model": model,
                "seeds": [None if seed is None else int(seed) for seed in seeds],
                "responses_path": str(responses_path),
                "metrics_path": str(metrics_path),
                "metadata_path": str(metadata_path),
                "metadata_models": metadata.get("models", []),
                "metadata_seeds": metadata.get("seeds", []),
                "temperature": metadata.get("temperature"),
                "max_new_tokens": metadata.get("max_new_tokens"),
                "api_base": metadata.get("api_base"),
                "api_key_env": metadata.get("api_key_env"),
                "records": int(metrics.shape[0]),
            }
        )

    combined = pd.concat(all_metrics, ignore_index=True)
    combined["adapter"] = args.adapter
    summary = _aggregate_metrics(combined, include_seed=True)

    summary_path = args.summary_output or (args.output_root / "summary.parquet")
    _save_table(summary, summary_path)

    seed_summary_path = (
        args.seed_summary_output
        or (args.output_root / "summary_by_model.parquet")
    )
    seed_summary = _aggregate_metrics(combined, include_seed=False)
    _save_table(seed_summary, seed_summary_path)

    manifest_path = args.output_root / "manifest.json"
    _ensure_parent(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "runs": manifests,
                "summary": str(summary_path),
                "seed_summary": str(seed_summary_path),
            },
            stream,
            indent=2,
        )

    logging.info("Benchmark complete. Summary at %s", summary_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

