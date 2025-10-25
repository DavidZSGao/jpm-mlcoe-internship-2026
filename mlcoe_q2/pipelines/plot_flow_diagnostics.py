"""Generate Li (2017)-style plots using flow diagnostics for Kernel flows.

This focuses on fast Kernel flows on CPU to avoid prohibitive EDH/LEDH costs.
Outputs:
- Figures under `reports/figures/`:
  - `li2017_{flow}_residuals.png` (before vs after)
  - `li2017_{flow}_movement.png`
  - `li2017_{flow}_logjac.png`
- A status index at `reports/q2/status/li2017_plots.md`
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mlcoe_q2.pipelines.benchmark import (
    _build_nonlinear_model,
    _simulate_sequence,
)
from mlcoe_q2.pipelines.flows import benchmark_flow
from mlcoe_q2.models.flows import KernelEmbeddedFlow
from mlcoe_q2.utils import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="*", type=int, default=[0], help="List of seeds")
    p.add_argument("--num-timesteps", type=int, default=15, help="Sequence length")
    p.add_argument("--particles", type=int, default=256, help="Number of particles")
    p.add_argument(
        "--outdir", type=Path, default=Path("reports/figures"), help="Figures dir"
    )
    p.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/li2017_plots.md"),
        help="Path to write Markdown status index",
    )
    add_config_argument(p)
    return parse_args_with_config(p, argv)


def _initial_particles(num_particles: int, state_dim: int, seed: int) -> tf.Tensor:
    tf.random.set_seed(seed)
    return tf.random.normal((num_particles, state_dim), dtype=tf.float32)


def _propagate(model, particles: tf.Tensor, seed: int) -> tf.Tensor:
    tf.random.set_seed(seed)
    num_particles = tf.shape(particles)[0]
    noise = tf.random.normal((num_particles, model.state_dim), dtype=tf.float32)
    cov_chol = tf.linalg.cholesky(model.process_noise_cov)
    noise = noise @ tf.transpose(cov_chol)

    def transition_fn(particle: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(model.transition_fn(particle, None), dtype=tf.float32)

    transitioned = tf.map_fn(transition_fn, particles, fn_output_signature=tf.float32)
    return transitioned + noise


def _run_kernel_flow_sequence(
    kernel_type: str,
    seeds: list[int],
    num_timesteps: int,
    num_particles: int,
):
    model = _build_nonlinear_model()
    sequences = {
        "mean_residual_before": [],
        "mean_residual_after": [],
        "mean_particle_movement": [],
        "log_jacobian_norm": [],
    }

    for seed in seeds:
        _, observations = _simulate_sequence(model, num_timesteps=num_timesteps, seed=seed)
        particles = _initial_particles(num_particles, model.state_dim, seed)
        weights = tf.fill((num_particles,), 1.0 / float(num_particles))

        seq_before = []
        seq_after = []
        seq_move = []
        seq_logj = []

        flow = KernelEmbeddedFlow(kernel_type=kernel_type, bandwidth=1.2, step_size=0.5, num_steps=3)

        for t in range(observations.shape[0]):
            particles = _propagate(model, particles, seed + t)
            obs_t = observations[t]

            bm = benchmark_flow(
                model=model,
                flow=flow,
                particles=particles,
                weights=weights,
                observation=obs_t,
                warmup=0,
                num_repeats=1,
            )
            diag = bm.diagnostics
            seq_before.append(diag["mean_residual_before"][-1])
            seq_after.append(diag["mean_residual_after"][-1])
            seq_move.append(diag["mean_particle_movement"][-1])
            seq_logj.append(diag["log_jacobian_norm"][-1])

            # Update particles/weights to the propagated ones for the next step
            particles = tf.convert_to_tensor(diag["propagated_particles"][-1], dtype=tf.float32)
            weights = tf.convert_to_tensor(diag["propagated_weights"][-1], dtype=tf.float32)

        sequences["mean_residual_before"].append(seq_before)
        sequences["mean_residual_after"].append(seq_after)
        sequences["mean_particle_movement"].append(seq_move)
        sequences["log_jacobian_norm"].append(seq_logj)

    # Aggregate over seeds
    def _agg(arrs):
        a = np.asarray(arrs, dtype=float)
        return a.mean(axis=0), a.std(axis=0)

    agg = {k: _agg(v) for k, v in sequences.items()}
    return agg


def _plot_series(x, mean, std, title, ylabel, out_path: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(x, mean, label="mean", color="#1f77b4")
    plt.fill_between(x, mean - std, mean + std, color="#1f77b4", alpha=0.2, label="±1σ")
    plt.title(title)
    plt.xlabel("timestep")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = list(args.seeds)

    flows = [
        ("KernelScalar", dict(kernel_type="scalar")),
        ("KernelDiagonal", dict(kernel_type="diagonal")),
        ("KernelMatrix", dict(kernel_type="matrix")),
    ]

    md_lines = ["# Li (2017)-Style Flow Diagnostics (Kernel)", ""]

    for name, cfg in flows:
        agg = _run_kernel_flow_sequence(
            kernel_type=cfg["kernel_type"],
            seeds=seeds,
            num_timesteps=args.num_timesteps,
            num_particles=args.particles,
        )
        x = np.arange(args.num_timesteps)

        # Residuals
        resid_mean = np.asarray(agg["mean_residual_before"][0])
        resid_std = np.asarray(agg["mean_residual_before"][1])
        resid_after_mean = np.asarray(agg["mean_residual_after"][0])
        resid_after_std = np.asarray(agg["mean_residual_after"][1])

        out1 = args.outdir / f"li2017_{name}_residuals.png"
        plt.figure(figsize=(7, 4))
        plt.plot(x, resid_mean, label="before", color="#d62728")
        plt.fill_between(x, resid_mean - resid_std, resid_mean + resid_std, color="#d62728", alpha=0.2)
        plt.plot(x, resid_after_mean, label="after", color="#2ca02c")
        plt.fill_between(x, resid_after_mean - resid_after_std, resid_after_mean + resid_after_std, color="#2ca02c", alpha=0.2)
        plt.title(f"{name}: residual norms before/after")
        plt.xlabel("timestep")
        plt.ylabel("residual norm")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out1.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out1, dpi=150)
        plt.close()

        # Movement
        move_mean, move_std = agg["mean_particle_movement"]
        out2 = args.outdir / f"li2017_{name}_movement.png"
        _plot_series(x, np.asarray(move_mean), np.asarray(move_std), f"{name}: particle movement", "movement", out2)

        # Log-Jacobian magnitude
        lj_mean, lj_std = agg["log_jacobian_norm"]
        out3 = args.outdir / f"li2017_{name}_logjac.png"
        _plot_series(x, np.asarray(lj_mean), np.asarray(lj_std), f"{name}: |log-Jacobian|", "|log-J|", out3)

        md_lines += [
            f"- **{name}**",
            f"  - Residuals: `{out1}`",
            f"  - Movement: `{out2}`",
            f"  - Log-Jacobian: `{out3}`",
        ]

    # Write status index
    args.status_md.parent.mkdir(parents=True, exist_ok=True)
    args.status_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
