"""Run multi-seed filter benchmarks and generate aggregate artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from mlcoe_q2.evaluation import aggregate_metrics, render_markdown_table
from mlcoe_q2.pipelines.benchmark import (
    _benchmark_differentiable_pf,
    _benchmark_ekf,
    _benchmark_particle_filter,
    _benchmark_ukf,
    _build_nonlinear_model,
    _simulate_sequence,
)
from mlcoe_q2.utils import add_config_argument, ensure_output_paths, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="List of seeds used for benchmarking",
    )
    parser.add_argument("--num-timesteps", type=int, default=15, help="Sequence length for simulation")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/artifacts"),
        help="Directory to write per-seed artifacts",
    )
    parser.add_argument(
        "--aggregate-out",
        type=Path,
        default=Path("reports/artifacts/benchmark_filters_multiseed.json"),
        help="Path to write aggregated JSON summary",
    )
    parser.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/filter_status.md"),
        help="Path to write Markdown status summary",
    )
    parser.add_argument("--log-level", default="INFO")
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


def _run_seed(seed: int, num_timesteps: int) -> dict:
    model = _build_nonlinear_model()
    _, observations = _simulate_sequence(model, num_timesteps=num_timesteps, seed=seed)

    pf = _benchmark_particle_filter(model, observations, seed)
    dpf = _benchmark_differentiable_pf(model, observations, seed)
    ekf = _benchmark_ekf(model, observations)
    ukf = _benchmark_ukf(model, observations)

    return {
        "seed": seed,
        "filter_results": {
            "PF": asdict(pf),
            "DifferentiablePF": asdict(dpf),
            "EKF": asdict(ekf),
            "UKF": asdict(ukf),
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
        path = args.outdir / f"benchmark_filters_seed_{s}.json"
        path.write_text(json.dumps(res, indent=2))
        logging.info("Wrote %s", path)

    metrics = ["runtime_s", "peak_memory_kb", "log_likelihood", "ess_mean"]
    aggregate = aggregate_metrics(per_seed, "filter_results", metrics)
    args.aggregate_out.write_text(json.dumps(aggregate.as_dict(), indent=2))
    logging.info("Wrote aggregate %s", args.aggregate_out)

    status_md = render_markdown_table(
        title="Q2 Filter Benchmark Status",
        aggregate=aggregate,
        metric_order=metrics,
        column_labels={
            "runtime_s": "Runtime (s)",
            "peak_memory_kb": "Peak Mem (KB)",
            "log_likelihood": "LogLik",
            "ess_mean": "Mean ESS",
        },
        notes=[
            "- DPF uses OT resampling; hyperparameters fixed for comparability",
        ],
    )
    ensure_output_paths([args.status_md])
    args.status_md.write_text(status_md)
    logging.info("Wrote status %s", args.status_md)


if __name__ == "__main__":  # pragma: no cover
    main()
