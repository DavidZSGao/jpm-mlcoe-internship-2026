"""Run multi-seed PF-PF with Dai (2022)-style stochastic flow parameter sweep vs LEDH.

Outputs (Q2 automation style):
- Per-seed JSON under `reports/artifacts/pfpf_dai22_seed_<seed>.json`
- Aggregated JSON at `reports/artifacts/pfpf_dai22_multiseed.json`
- Markdown status at `reports/q2/status/pfpf_dai22.md`
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
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4], help="Seeds")
    p.add_argument("--num-timesteps", type=int, default=15, help="Sequence length")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/artifacts"),
        help="Per-seed and aggregate output directory",
    )
    p.add_argument(
        "--aggregate-out",
        type=Path,
        default=Path("reports/artifacts/pfpf_dai22_multiseed.json"),
        help="Aggregate JSON output path",
    )
    p.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/pfpf_dai22.md"),
        help="Status Markdown output path",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)


def _run_seed(seed: int, num_timesteps: int) -> dict:
    model = _build_nonlinear_model()
    _, observations = _simulate_sequence(model, num_timesteps=num_timesteps, seed=seed)

    methods = {
        "PF_PF_LEDH": lambda: LocalExactDaumHuangFlow(step_size=0.8, num_steps=3),
        # Dai(2022)-style: smaller step size, more steps, varying diffusion
        "PF_PF_SPF_A": lambda: StochasticParticleFlow(step_size=0.6, num_steps=8, diffusion=0.05),
        "PF_PF_SPF_B": lambda: StochasticParticleFlow(step_size=0.6, num_steps=8, diffusion=0.10),
        "PF_PF_SPF_C": lambda: StochasticParticleFlow(step_size=0.4, num_steps=12, diffusion=0.15),
    }

    results = {}
    for name, make_flow in methods.items():
        res = _benchmark_pfpf(model, observations, seed, make_flow)
        results[name] = asdict(res)

    return {
        "seed": seed,
        "pfpf_results": results,
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
        "# Q2 PF-PF (Dai 2022) Parameter Sweep",
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
        "- Stochastic Particle Flow configurations approximate Dai (2022): smaller step-size, more steps, variable diffusion",
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
        path = args.outdir / f"pfpf_dai22_seed_{s}.json"
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
