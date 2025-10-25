"""Run multi-seed DPF comparisons across soft, OT-low, and OT resampling."""

from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import tensorflow as tf

from mlcoe_q2.evaluation import aggregate_metrics, render_markdown_table
from mlcoe_q2.models.filters import differentiable_particle_filter
from mlcoe_q2.pipelines.benchmark import (
    _build_nonlinear_model,
    _initial_particles,
    _simulate_sequence,
)
from mlcoe_q2.utils import add_config_argument, ensure_output_paths, parse_args_with_config


ess_key = "ess_mean"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="Seeds")
    parser.add_argument("--num-timesteps", type=int, default=15, help="Sequence length")
    parser.add_argument("--particles", type=int, default=256, help="Number of particles")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/artifacts"),
        help="Directory to write per-seed artifacts",
    )
    parser.add_argument(
        "--aggregate-out",
        type=Path,
        default=Path("reports/artifacts/dpf_comparisons_multiseed.json"),
        help="Path to write aggregated JSON summary",
    )
    parser.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/dpf_comparisons.md"),
        help="Path to write Markdown status summary",
    )
    parser.add_argument("--log-level", default="INFO")
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


def _time_and_profile(fn):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end - start, peak_mem / 1024.0


def _run_variant(model, observations, seed: int, particles: int, method: str) -> dict:
    init_parts = _initial_particles(particles, model.state_dim, seed)

    def run():
        return differentiable_particle_filter(
            model=model,
            observations=observations,
            num_particles=particles,
            initial_particles=init_parts,
            mix_with_uniform=0.15,
            ot_epsilon=0.25,
            ot_num_iters=25,
            resampling_method=method,
        )

    res, runtime, peak_kb = _time_and_profile(run)
    ess = tf.reduce_mean(res.diagnostics["ess"]).numpy()
    out = {
        "runtime_s": float(runtime),
        "peak_memory_kb": float(peak_kb),
        "log_likelihood": float(res.log_likelihood.numpy()),
        ess_key: float(ess),
    }
    return out


def _run_seed(seed: int, num_timesteps: int, particles: int) -> dict:
    model = _build_nonlinear_model()
    _, observations = _simulate_sequence(model, num_timesteps=num_timesteps, seed=seed)

    soft = _run_variant(model, observations, seed, particles, method="soft")
    ot_low = _run_variant(model, observations, seed, particles, method="ot_low")
    ot = _run_variant(model, observations, seed, particles, method="ot")

    return {
        "seed": seed,
        "dpf_results": {
            "DPF_Soft": soft,
            "DPF_OT_Low": ot_low,
            "DPF_OT": ot,
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    ensure_output_paths([args.outdir, args.aggregate_out, args.status_md])

    per_seed: list[dict] = []
    for s in args.seeds:
        logging.info("Running seed %d", s)
        res = _run_seed(seed=s, num_timesteps=args.num_timesteps, particles=args.particles)
        per_seed.append(res)
        path = args.outdir / f"dpf_comparisons_seed_{s}.json"
        path.write_text(json.dumps(res, indent=2))
        logging.info("Wrote %s", path)

    metrics = ["runtime_s", "peak_memory_kb", "log_likelihood", ess_key]
    aggregate = aggregate_metrics(per_seed, "dpf_results", metrics)
    args.aggregate_out.write_text(json.dumps(aggregate.as_dict(), indent=2))
    logging.info("Wrote aggregate %s", args.aggregate_out)

    status_md = render_markdown_table(
        title="Q2 DPF Resampling Comparisons",
        aggregate=aggregate,
        metric_order=metrics,
        column_labels={
            "runtime_s": "Runtime (s)",
            "peak_memory_kb": "Peak Mem (KB)",
            "log_likelihood": "LogLik",
            ess_key: "Mean ESS",
        },
        notes=[
            "- Soft weights (no transport), OT low-iter, and full OT",
        ],
    )
    ensure_output_paths([args.status_md])
    args.status_md.write_text(status_md)
    logging.info("Wrote status %s", args.status_md)


if __name__ == "__main__":  # pragma: no cover
    main()
