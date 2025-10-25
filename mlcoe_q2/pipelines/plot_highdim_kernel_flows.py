"""Hu (2021)-style high-dimensional diagnostics for Kernel flows.

Generates residual/movement/|log-J| plots and an observed-marginal variance
plot to illustrate collapse prevention by matrix-valued kernels in higher dims.

Outputs under reports/:
- Figures: reports/figures/hu2021_*.png
- Status MD: reports/q2/status/hu2021_plots.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.pipelines.flows import benchmark_flow
from mlcoe_q2.models.flows import KernelEmbeddedFlow
from mlcoe_q2.utils import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="*", type=int, default=[0], help="List of seeds")
    p.add_argument("--num-timesteps", type=int, default=10, help="Sequence length")
    p.add_argument("--particles", type=int, default=256, help="Number of particles")
    p.add_argument("--state-dim", type=int, default=16, help="State dimension")
    p.add_argument("--obs-dim", type=int, default=4, help="Observed dimension")
    p.add_argument(
        "--outdir", type=Path, default=Path("reports/figures"), help="Figures dir"
    )
    p.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/hu2021_plots.md"),
        help="Path to write Markdown status index",
    )
    add_config_argument(p)
    return parse_args_with_config(p, argv)


def build_highdim_model(state_dim: int, obs_dim: int) -> NonlinearStateSpaceModel:
    assert obs_dim <= state_dim
    # Mildly coupled nonlinear transition.
    proc = 0.02 * tf.eye(state_dim, dtype=tf.float32)

    def transition_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        x = tf.convert_to_tensor(state, dtype=tf.float32)
        # Coupled ring: x_i' = a * x_i + b * tanh(x_{i-1})
        a = 0.94
        b = 0.08
        x_roll = tf.concat([x[-1:], x[:-1]], axis=0)
        return a * x + b * tf.math.tanh(x_roll)

    # Observe first obs_dim with small linear mix to couple signals.
    obs_cov = 0.1 * tf.eye(obs_dim, dtype=tf.float32)

    W = tf.concat(
        [
            tf.eye(obs_dim, dtype=tf.float32),
            tf.zeros((obs_dim, state_dim - obs_dim), dtype=tf.float32),
        ],
        axis=1,
    )
    M = W + 0.05 * tf.random.uniform(W.shape, minval=-1.0, maxval=1.0, dtype=tf.float32)

    def observation_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        x = tf.convert_to_tensor(state, dtype=tf.float32)
        y = tf.linalg.matvec(M, x)
        # small nonlinearity on observed components
        return y + 0.05 * tf.math.sin(y)

    return NonlinearStateSpaceModel(
        state_dim=state_dim,
        observation_dim=obs_dim,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        process_noise_cov=proc,
        observation_noise_cov=obs_cov,
    )


def simulate(model: NonlinearStateSpaceModel, num_timesteps: int, seed: int):
    tf.random.set_seed(seed)
    init = tf.zeros((model.state_dim,), dtype=tf.float32)
    states, obs = model.simulate(num_timesteps=num_timesteps, initial_state=init, seed=seed)
    return states, obs


def initial_particles(num_particles: int, state_dim: int, seed: int) -> tf.Tensor:
    tf.random.set_seed(seed)
    return tf.random.normal((num_particles, state_dim), dtype=tf.float32)


def observed_variance(particles: tf.Tensor, obs_mat: tf.Tensor, obs_dim: int) -> float:
    # Approximate marginal variance in observed space using the linear map 'obs_mat'.
    Y = tf.matmul(tf.convert_to_tensor(particles, dtype=tf.float32), obs_mat, transpose_b=True)
    Y = Y[:, :obs_dim]
    var = tf.math.reduce_variance(Y, axis=0)
    return float(tf.reduce_mean(var).numpy())


def plot_series(x, mean, std, title, ylabel, out_path: Path):
    plt.figure(figsize=(7, 4))
    mean = np.asarray(mean)
    std = np.asarray(std)
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


def run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = list(args.seeds)
    model = build_highdim_model(args.state_dim, args.obs_dim)

    # Reuse the observation linear part M for observed variance.
    # We reconstruct M from observation_fn closure by re-building with same seed.
    # For simplicity, rebuild M here identically as in build_highdim_model.
    W = tf.concat(
        [tf.eye(args.obs_dim, dtype=tf.float32), tf.zeros((args.obs_dim, args.state_dim - args.obs_dim), dtype=tf.float32)],
        axis=1,
    )
    M = W + 0.05 * tf.random.uniform(W.shape, minval=-1.0, maxval=1.0, dtype=tf.float32)

    flow_cfgs = [
        ("KernelScalar", dict(kernel_type="scalar")),
        ("KernelDiagonal", dict(kernel_type="diagonal")),
        ("KernelMatrix", dict(kernel_type="matrix")),
    ]

    md_lines = ["# Hu (2021)-Style High-Dimensional Diagnostics (Kernel)", ""]

    for name, cfg in flow_cfgs:
        series = {k: [] for k in [
            "mean_residual_before", "mean_residual_after", "mean_particle_movement", "log_jacobian_norm", "obs_marginal_variance"
        ]}

        for seed in seeds:
            _, obs = simulate(model, num_timesteps=args.num_timesteps, seed=seed)
            parts = initial_particles(args.particles, model.state_dim, seed)
            weights = tf.fill((args.particles,), 1.0 / float(args.particles))

            flow = KernelEmbeddedFlow(kernel_type=cfg["kernel_type"], bandwidth=1.2, step_size=0.4, num_steps=3)

            seq_before = []
            seq_after = []
            seq_move = []
            seq_logj = []
            seq_var = []

            for t in range(obs.shape[0]):
                # simple propagation with process noise
                # re-use plot_flow_diagnostics propagation
                num_particles = tf.shape(parts)[0]
                noise = tf.random.normal((num_particles, model.state_dim), dtype=tf.float32)
                cov_chol = tf.linalg.cholesky(model.process_noise_cov)
                noise = noise @ tf.transpose(cov_chol)

                def trans_fn(p: tf.Tensor) -> tf.Tensor:
                    return tf.convert_to_tensor(model.transition_fn(p, None), dtype=tf.float32)

                parts = tf.map_fn(trans_fn, parts, fn_output_signature=tf.float32) + noise

                bm = benchmark_flow(
                    model=model,
                    flow=flow,
                    particles=parts,
                    weights=weights,
                    observation=obs[t],
                    warmup=0,
                    num_repeats=1,
                )
                diag = bm.diagnostics
                seq_before.append(diag["mean_residual_before"][-1])
                seq_after.append(diag["mean_residual_after"][-1])
                seq_move.append(diag["mean_particle_movement"][-1])
                seq_logj.append(diag["log_jacobian_norm"][-1])

                parts = tf.convert_to_tensor(diag["propagated_particles"][-1], dtype=tf.float32)
                weights = tf.convert_to_tensor(diag["propagated_weights"][-1], dtype=tf.float32)

                seq_var.append(observed_variance(parts, M, args.obs_dim))

            series["mean_residual_before"].append(seq_before)
            series["mean_residual_after"].append(seq_after)
            series["mean_particle_movement"].append(seq_move)
            series["log_jacobian_norm"].append(seq_logj)
            series["obs_marginal_variance"].append(seq_var)

        def agg(arrs):
            a = np.asarray(arrs, dtype=float)
            return a.mean(axis=0), a.std(axis=0)

        x = np.arange(args.num_timesteps)

        for key, (ylabel, fname) in {
            "mean_residual_before": ("residual norm", f"hu2021_{name}_residuals_before.png"),
            "mean_residual_after": ("residual norm", f"hu2021_{name}_residuals_after.png"),
            "mean_particle_movement": ("movement", f"hu2021_{name}_movement.png"),
            "log_jacobian_norm": ("|log-J|", f"hu2021_{name}_logjac.png"),
            "obs_marginal_variance": ("observed marginal variance", f"hu2021_{name}_obsvar.png"),
        }.items():
            mean, std = agg(series[key])
            outp = args.outdir / fname
            plot_series(x, mean, std, f"{name}: {ylabel}", ylabel, outp)

        md_lines += [
            f"- **{name}**",
            f"  - Residual (before): `reports/figures/hu2021_{name}_residuals_before.png`",
            f"  - Residual (after): `reports/figures/hu2021_{name}_residuals_after.png`",
            f"  - Movement: `reports/figures/hu2021_{name}_movement.png`",
            f"  - Log-Jacobian: `reports/figures/hu2021_{name}_logjac.png`",
            f"  - Observed variance: `reports/figures/hu2021_{name}_obsvar.png`",
        ]

    args.status_md.parent.mkdir(parents=True, exist_ok=True)
    args.status_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    run()
