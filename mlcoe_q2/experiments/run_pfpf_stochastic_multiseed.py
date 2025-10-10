"""Run multi-seed PF-PF benchmarks comparing LEDH vs Stochastic flow proposals.

Outputs (imitates Q1/Q2 automation style):
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
from typing import Iterable, Sequence

import numpy as np

from mlcoe_q2.experiments.benchmark import (
    _build_nonlinear_model,
    _simulate_sequence,
    _benchmark_pfpf,
)
from mlcoe_q2.flows import LocalExactDaumHuangFlow, StochasticParticleFlow


ess_key = "ess_mean"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="List of seeds")
    p.add_argument("--num-timesteps", type=int, default=15, help="Sequence length")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/artifacts"),
        help="Directory to write per-seed artifacts and aggregates",
    )
    p.add_argument(
        "--aggregate-out",
        type=Path,
        default=Path("reports/artifacts/pfpf_stochastic_multiseed.json"),
        help="Path to write aggregated JSON summary",
    )
    p.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/pfpf_status.md"),
        help="Path to write Markdown status summary",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True) if p.suffix else p.mkdir(parents=True, exist_ok=True)


def _run_seed(seed: int, num_timesteps: int) -> dict:
    model = _build_nonlinear_model()
    _, observations = _simulate_sequence(model, num_timesteps=num_timesteps, seed=seed)

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


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None}
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0))}


def _aggregate(per_seed: list[dict]) -> dict:
    methods = list(per_seed[0]["pfpf_results"].keys()) if per_seed else []
    metrics = ["runtime_s", "peak_memory_kb", "log_likelihood", ess_key]

    out = {"num_seeds": len(per_seed), "metrics": {}}
    for metric in metrics:
        metric_block = {}
        for m in methods:
            vals = []
            for r in per_seed:
                v = r["pfpf_results"][m].get(metric)
                if v is not None:
                    vals.append(v)
            metric_block[m] = _agg(vals)
        out["metrics"][metric] = metric_block
    return out


def _render_status_md(agg: dict) -> str:
    hdr = ["Method", "Runtime (s)", "Peak Mem (KB)", "LogLik", "Mean ESS"]
    methods = list(next(iter(agg["metrics"].values())).keys()) if agg["metrics"] else []

    def cell(metric: str, m: str) -> str:
        d = agg["metrics"][metric][m]
        if d["mean"] is None:
            return "—"
        return f"{d['mean']:.2f} ± {d['std']:.2f}"

    lines = ["# Q2 PF-PF Benchmark Status", "", "| " + " | ".join(hdr) + " |", "| " + " | ".join(["---"] * len(hdr)) + " |"]
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
        "- LEDH vs Stochastic flow used as PF-PF proposals",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    _ensure_dirs([args.outdir, args.aggregate_out, args.status_md])

    per_seed: list[dict] = []
    for s in args.seeds:
        logging.info("Running seed %d", s)
        res = _run_seed(seed=s, num_timesteps=args.num_timesteps)
        per_seed.append(res)
        path = args.outdir / f"pfpf_stochastic_seed_{s}.json"
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
