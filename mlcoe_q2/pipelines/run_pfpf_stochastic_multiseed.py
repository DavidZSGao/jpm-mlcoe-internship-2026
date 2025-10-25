"""Run multi-seed PF-PF benchmarks comparing LEDH and stochastic flow proposals.

Outputs (mirrors the Q1 automation style):
- Per-seed JSON under `reports/artifacts/pfpf_stochastic_seed_<seed>.json`
- Aggregated JSON at `reports/artifacts/pfpf_stochastic_multiseed.json`
- Markdown status at `reports/q2/status/pfpf_status.md`
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from mlcoe_q2.evaluation import aggregate_metrics, render_markdown_table
from mlcoe_q2.models.flows import LocalExactDaumHuangFlow, StochasticParticleFlow
from mlcoe_q2.pipelines.benchmark import (
    _benchmark_pfpf,
    _build_nonlinear_model,
    _simulate_sequence,
)
from mlcoe_q2.utils import add_config_argument, ensure_output_paths, parse_args_with_config


ess_key = "ess_mean"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="List of seeds",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=15,
        help="Sequence length for simulation",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/artifacts"),
        help="Directory to write per-seed artifacts",
    )
    parser.add_argument(
        "--aggregate-out",
        type=Path,
        default=Path("reports/artifacts/pfpf_stochastic_multiseed.json"),
        help="Path to write aggregated JSON summary",
    )
    parser.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/pfpf_status.md"),
        help="Path to write Markdown status summary",
    )
    parser.add_argument("--log-level", default="INFO")
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


def _run_seed(seed: int, num_timesteps: int) -> dict:
    model = _build_nonlinear_model()
    _, observations = _simulate_sequence(
        model,
        num_timesteps=num_timesteps,
        seed=seed,
    )

    ledh = _benchmark_pfpf(
        model,
        observations,
        seed,
        lambda: LocalExactDaumHuangFlow(step_size=0.8, num_steps=3),
    )
    stochastic = _benchmark_pfpf(
        model,
        observations,
        seed,
        lambda: StochasticParticleFlow(step_size=0.8, num_steps=6, diffusion=0.08),
    )

    return {
        "seed": seed,
        "pfpf_results": {
            "PF_PF_LEDH": asdict(ledh),
            "PF_PF_Stochastic": asdict(stochastic),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    ensure_output_paths([args.outdir, args.aggregate_out, args.status_md])

    per_seed: list[dict] = []
    for s in args.seeds:
        logging.info("Running seed %d", s)
        res = _run_seed(seed=s, num_timesteps=args.num_timesteps)
        per_seed.append(res)
        path = args.outdir / f"pfpf_stochastic_seed_{s}.json"
        path.write_text(json.dumps(res, indent=2))
        logging.info("Wrote %s", path)

    metrics = ["runtime_s", "peak_memory_kb", "log_likelihood", ess_key]
    aggregate = aggregate_metrics(per_seed, "pfpf_results", metrics)
    args.aggregate_out.write_text(json.dumps(aggregate.as_dict(), indent=2))
    logging.info("Wrote aggregate %s", args.aggregate_out)

    status_md = render_markdown_table(
        title="Q2 PF-PF Benchmark Status",
        aggregate=aggregate,
        metric_order=metrics,
        column_labels={
            "runtime_s": "Runtime (s)",
            "peak_memory_kb": "Peak Mem (KB)",
            "log_likelihood": "LogLik",
            ess_key: "Mean ESS",
        },
        notes=[
            "- LEDH and stochastic proposals share PF-PF hyperparameters",
        ],
    )
    ensure_output_paths([args.status_md])
    args.status_md.write_text(status_md)
    logging.info("Wrote status %s", args.status_md)


if __name__ == "__main__":  # pragma: no cover
    main()
