"""Run a registered LLM adapter over the prompt dataset and capture responses."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from mlcoe_q1.llm.adapters import create_adapter
from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument("--prompts", type=Path, required=True, help="Prompt dataset path")
    parser.add_argument("--output", type=Path, required=True, help="Where to store responses")
    parser.add_argument(
        "--adapter",
        default="flan-t5",
        help="Adapter registry name (e.g. flan-t5)",
    )
    parser.add_argument("--model", help="Optional model identifier override")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of prompts to process")
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated random seeds to evaluate",
    )
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
        help="Environment variable holding the API key for hosted adapters",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Request timeout (seconds) for hosted adapters",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional metadata file path (defaults to <output>.metadata.json)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "prompts": Path,
            "output": Path,
            "metadata_output": Path,
        },
    )


def _build_prompt(row: pd.Series) -> str:
    statements = [str(x) for x in row["statements"]]
    schema = json.dumps({name: {} for name in statements}, indent=2)
    parts = [
        "You are a financial modelling assistant.",
        "Output valid JSON only (no code fences). Use JSON numbers without commas or underscores.",
        "Respond with JSON only using the structure shown below.",
        schema,
        str(row["prompt"]),
    ]
    return "\n\n".join(parts)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\\s*(\\{.*\\})\\s*```", stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()
        else:
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]
    return stripped


def _remove_numeric_group_separators(text: str) -> str:
    parts = text.split('"')
    for idx in range(0, len(parts), 2):
        segment = parts[idx]
        segment = re.sub(r"(?<=\\d)[,_](?=\\d{3}(\\D|$))", "", segment)
        segment = re.sub(r"(?<=\\d)\\s+(?=\\d)", "", segment)
        parts[idx] = segment
    return '"'.join(parts)


def _parse_json_payload(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        cleaned = _strip_code_fences(text)
        cleaned = _remove_numeric_group_separators(cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = {}
    if isinstance(payload, dict):
        return payload
    return {}


def _fallback_parse_payload(
    text: str,
    statements: Sequence[str],
) -> dict[str, dict[str, object]]:
    cleaned = _strip_code_fences(text)
    cleaned = _remove_numeric_group_separators(cleaned)
    payload: dict[str, dict[str, object]] = {}
    number_pattern = re.compile(r'"([^"]+)"\s*:\s*(-?\d[\d,_.\s]*)')
    for statement in statements:
        start = cleaned.find(f'"{statement}"')
        if start == -1:
            continue
        brace_start = cleaned.find("{", start)
        if brace_start == -1:
            continue
        if statement == "balance_sheet":
            next_start = cleaned.find('"income_statement"', brace_start)
            section_text = cleaned[brace_start + 1 : next_start] if next_start != -1 else cleaned[brace_start + 1 :]
        else:
            section_text = cleaned[brace_start + 1 :]
        entries: dict[str, object] = {}
        for match in number_pattern.finditer(section_text):
            key = match.group(1)
            raw_value = match.group(2)
            normalized = re.sub(r"[\s,_]", "", raw_value)
            try:
                value = float(normalized)
            except ValueError:
                continue
            entries[key] = value
        if entries:
            payload[str(statement)] = entries
    return payload


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _canonicalize_section(
    section: dict[str, object],
    allowed_keys: Sequence[str],
) -> dict[str, object]:
    normalized_map: dict[str, str] = {}
    collisions: set[str] = set()
    for key in allowed_keys:
        normalized = _normalize_key(key)
        if normalized in normalized_map and normalized_map[normalized] != key:
            collisions.add(normalized)
        else:
            normalized_map[normalized] = key
    for normalized in collisions:
        normalized_map.pop(normalized, None)

    remapped: dict[str, object] = {}
    for key, value in section.items():
        key_str = str(key)
        normalized = _normalize_key(key_str)
        canonical = normalized_map.get(normalized, key_str)
        remapped[canonical] = value
    return remapped


def _coerce_payload(
    text: str,
    statements: Sequence[str],
    allowed_keys: dict[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, object]]:
    payload = _parse_json_payload(text)
    if not payload:
        payload = _fallback_parse_payload(text, statements)

    result: dict[str, dict[str, object]] = {}
    for name in statements:
        section = payload.get(name, {}) if isinstance(payload, dict) else {}
        if not isinstance(section, dict):
            section = {}
        if allowed_keys and name in allowed_keys:
            section = _canonicalize_section(section, allowed_keys[name])
        cleaned: dict[str, object] = {}
        for key, value in section.items():
            cleaned[str(key)] = value
        result[str(name)] = cleaned
    return result


def _collect_models(responses: pd.DataFrame) -> list[str]:
    if "model" not in responses.columns:
        return []
    models = {
        str(value)
        for value in responses["model"].dropna().unique().tolist()
        if str(value)
    }
    return sorted(models)


def _collect_seeds(responses: pd.DataFrame) -> list[int | None]:
    if "seed" not in responses.columns:
        return []
    seeds: set[int | None] = set()
    for value in responses["seed"].unique().tolist():
        if pd.isna(value):
            seeds.add(None)
        else:
            seeds.add(int(value))
    return sorted(
        seeds,
        key=lambda item: (item is None, item if item is not None else 0),
    )


def _resolve_metadata_path(output: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    return output.with_name(output.name + ".metadata.json")


def build_metadata(
    *,
    adapter: str,
    prompts_path: Path,
    output_path: Path,
    temperature: float,
    max_new_tokens: int,
    limit: int,
    api_base: str | None,
    api_key_env: str | None,
    request_timeout: float,
    responses: pd.DataFrame,
    model_argument: str | None = None,
    seed_argument: str | None = None,
) -> dict[str, object]:
    models = _collect_models(responses)
    seeds = _collect_seeds(responses)
    uses_hosted_api = adapter.lower() in {"openai", "openai-chat"} or bool(api_base)
    metadata: dict[str, object] = {
        "adapter": adapter,
        "models": models,
        "seeds": seeds,
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "limit": int(limit),
        "records": int(len(responses)),
        "prompts_path": str(prompts_path),
        "output_path": str(output_path),
        "api_base": api_base if uses_hosted_api else None,
        "api_key_env": api_key_env if uses_hosted_api else None,
        "request_timeout": float(request_timeout),
    }
    if model_argument:
        metadata["model_argument"] = model_argument
    if seed_argument:
        metadata["seed_argument"] = seed_argument
    return metadata


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    prompts = pd.read_parquet(args.prompts)
    if args.limit > 0:
        prompts = prompts.head(args.limit)

    if args.model:
        raw_models = [token.strip() for token in args.model.split(',') if token.strip()]
        model_ids: list[str | None] = raw_models or [args.model.strip()]
    else:
        model_ids = [None]

    seed_values: list[int | None] = []
    if args.seeds:
        for token in args.seeds.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                seed_values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid seed value: {token}") from exc
    if not seed_values:
        seed_values = [None]

    records: list[dict[str, object]] = []

    for model_id in model_ids:
        adapter = create_adapter(
            args.adapter,
            model=model_id,
            model_name=model_id,
            max_new_tokens=args.max_new_tokens,
            api_base=args.api_base or None,
            api_key_env=args.api_key_env,
            request_timeout=args.request_timeout,
        )

        for seed in seed_values:
            if seed is not None:
                if hasattr(adapter, "set_seed"):
                    try:
                        adapter.set_seed(seed)
                    except Exception as exc:  # pragma: no cover - defensive
                        logging.warning(
                            "Unable to set seed %s for %s: %s",
                            seed,
                            adapter.model_id,
                            exc,
                        )
                else:
                    logging.warning(
                        "Adapter %s does not support seeding", adapter.name
                    )

            for row in prompts.itertuples(index=False):
                row_series = pd.Series(row._asdict())
                prompt_text = _build_prompt(row_series)
                statements = [str(x) for x in row_series["statements"]]
                allowed_keys: dict[str, Sequence[str]] | None = None
                if "context" in row_series:
                    try:
                        context_payload = json.loads(row_series["context"])
                    except (TypeError, json.JSONDecodeError):
                        context_payload = None
                    if isinstance(context_payload, dict):
                        allowed_keys = {
                            str(statement): list(payload.keys())
                            for statement, payload in context_payload.items()
                            if isinstance(payload, dict)
                        }
                completion = adapter.generate(
                    prompt_text,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
                payload = _coerce_payload(completion, statements, allowed_keys)
                record = {k: getattr(row, k) for k in ["ticker", "context_period", "target_period"]}
                record.update(
                    {
                        "model": adapter.model_id,
                        "adapter": adapter.name,
                        "seed": seed,
                        "response": json.dumps(payload),
                        "raw_response": completion,
                    }
                )
                records.append(record)

    output_df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output, index=False)
    logging.info("Wrote %d responses to %s", len(output_df), args.output)

    metadata_path = _resolve_metadata_path(args.output, args.metadata_output)
    metadata = build_metadata(
        adapter=args.adapter,
        prompts_path=args.prompts,
        output_path=args.output,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        api_base=args.api_base or None,
        api_key_env=args.api_key_env,
        request_timeout=args.request_timeout,
        responses=output_df,
        model_argument=args.model,
        seed_argument=args.seeds or None,
    )
    _write_metadata(metadata_path, metadata)
    logging.info("Wrote metadata to %s", metadata_path)


if __name__ == "__main__":
    main()
