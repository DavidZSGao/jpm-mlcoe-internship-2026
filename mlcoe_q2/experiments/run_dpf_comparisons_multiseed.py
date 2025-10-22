"""Run multi-seed DPF comparisons: soft vs OT-low vs OT resampling.

Outputs (Q2 automation style):
- Per-seed JSON under `reports/artifacts/dpf_comparisons_seed_<seed>.json`
- Aggregated JSON at `reports/artifacts/dpf_comparisons_multiseed.json`
- Markdown status at `reports/q2/status/dpf_comparisons.md`
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf

from mlcoe_q2.experiments.benchmark import (
    _build_nonlinear_model,
    _simulate_sequence,
    _initial_particles,
)
from mlcoe_q2.filters import differentiable_particle_filter


ess_key = "ess_mean"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="Seeds")
    p.add_argument("--num-timesteps", type=int, default=15, help="Sequence length")
    p.add_argument("--particles", type=int, default=256, help="Number of particles")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/artifacts"),
        help="Directory to write per-seed artifacts and aggregates",
    )
    p.add_argument(
        "--aggregate-out",
        type=Path,
        default=Path("reports/artifacts/dpf_comparisons_multiseed.json"),
        help="Path to write aggregated JSON summary",
    )
    p.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/dpf_comparisons.md"),
        help="Path to write Markdown status summary",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)


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


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None}
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0))}


def _aggregate(per_seed: list[dict]) -> dict:
    methods = list(per_seed[0]["dpf_results"].keys()) if per_seed else []
    metrics = ["runtime_s", "peak_memory_kb", "log_likelihood", ess_key]

    out = {"num_seeds": len(per_seed), "metrics": {}}
    for metric in metrics:
        metric_block = {}
        for m in methods:
            vals = []
            for r in per_seed:
                v = r["dpf_results"][m].get(metric)
                if v is not None:
                    vals.append(v)
            metric_block[m] = _agg(vals)
        out["metrics"][metric] = metric_block
    return out


def _render_status_md(agg: dict) -> str:
    hdr = ["Method", "Runtime (s)", "Peak Mem (KB)", "LogLik", "Mean ESS"]
    methods: list[str] = []
    if agg["metrics"]:
        first = next(iter(agg["metrics"].values()))
        methods = list(first.keys())

    def cell(metric: str, m: str) -> str:
        d = agg["metrics"][metric][m]
        if d["mean"] is None:
            return "—"
        return f"{d['mean']:.2f} ± {d['std']:.2f}"

    lines = [
        "# Q2 DPF Resampling Comparisons",
        "",
        "| " + " | ".join(hdr) + " |",
        "| " + " | ".join(["---"] * len(hdr)) + " |",
    ]
    for m in methods:
        row = [
            m,
            cell("runtime_s", m),
            cell("peak_memory_kb", m),
            cell("log_likelihood", m),
            cell(ess_key, m),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Notes",
        "- Aggregated across multiple seeds",
        "- Methods: soft weights (no transport), OT low-iter, and full OT",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    _ensure_dirs([args.outdir, args.aggregate_out, args.status_md])

    per_seed: list[dict] = []
    for s in args.seeds:
        logging.info("Running seed %d", s)
        res = _run_seed(seed=s, num_timesteps=args.num_timesteps, particles=args.particles)
        per_seed.append(res)
        path = args.outdir / f"dpf_comparisons_seed_{s}.json"
        path.write_text(json.dumps(res, indent=2))
        logging.info("Wrote %s", path)

    agg = _aggregate(per_seed)
    args.aggregate_out.write_text(json.dumps(agg, indent=2))
    logging.info("Wrote aggregate %s", args.aggregate_out)

    md = _render_status_md(agg)
    args.status_md.parent.mkdir(parents=True, exist_ok=True)
    args.status_md.write_text(md)
    logging.info("Wrote status %s", args.status_md)


if __name__ == "__main__":  # pragma: no cover
    main()
