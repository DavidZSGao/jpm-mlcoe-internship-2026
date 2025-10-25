
"""Diagnostics comparing nonlinear filters on the benchmark SSM."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.models.filters import (
    differentiable_particle_filter,
    extended_kalman_filter,
    particle_filter,
    unscented_kalman_filter,
)
from mlcoe_q2.utils import add_config_argument, parse_args_with_config


@dataclasses.dataclass
class FilterDiagnostics:
    """Single-seed diagnostics for a filter."""

    rmse: float
    rmse_by_dim: np.ndarray
    rmse_per_timestep: np.ndarray
    log_likelihood: Optional[float]
    ess_min: Optional[float]
    ess_mean: Optional[float]
    ess_ratio_min: Optional[float]
    ess_ratio_mean: Optional[float]
    ess_ratio_sequence: Optional[np.ndarray]
    runtime_s: float
    notes: Optional[str]


@dataclasses.dataclass
class AggregateDiagnostics:
    """Diagnostics aggregated across seeds for a filter."""

    rmse_mean: float
    rmse_std: float
    rmse_by_dim_mean: list[float]
    rmse_by_dim_std: list[float]
    rmse_per_timestep_mean: list[float]
    rmse_per_timestep_std: list[float]
    log_likelihood_mean: Optional[float]
    log_likelihood_std: Optional[float]
    ess_min_mean: Optional[float]
    ess_min_std: Optional[float]
    ess_mean_mean: Optional[float]
    ess_mean_std: Optional[float]
    ess_ratio_min_mean: Optional[float]
    ess_ratio_min_std: Optional[float]
    ess_ratio_mean_mean: Optional[float]
    ess_ratio_mean_std: Optional[float]
    ess_ratio_sequence_mean: Optional[list[float]]
    ess_ratio_sequence_std: Optional[list[float]]
    runtime_mean: float
    runtime_std: float
    notes: Optional[str]
    per_seed: list[FilterDiagnostics]


def _build_nonlinear_model() -> NonlinearStateSpaceModel:
    transition_cov = tf.constant([[0.05, 0.0], [0.0, 0.03]], dtype=tf.float32)
    observation_cov = tf.constant([[0.2]], dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def transition_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        x0 = state[0]
        x1 = state[1]
        new_x0 = 0.85 * x0 + 0.25 * tf.math.sin(x1) + 0.05 * tf.math.sin(3.0 * x0)
        new_x1 = 0.9 * x1 + 0.2 * tf.math.tanh(x0)
        return tf.stack([new_x0, new_x1])

    @tf.function(reduce_retracing=True)
    def observation_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        return tf.stack([
            0.6 * tf.math.sin(state[0])
            + 0.1 * state[1]
            + 0.05 * tf.math.sin(2.5 * state[0])
        ])

    return NonlinearStateSpaceModel(
        state_dim=2,
        observation_dim=1,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        process_noise_cov=transition_cov,
        observation_noise_cov=observation_cov,
    )


def _simulate_sequence(
    model: NonlinearStateSpaceModel,
    num_timesteps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    tf.random.set_seed(seed)
    initial_state = tf.constant([0.6, -0.4], dtype=tf.float32)
    states, observations = model.simulate(
        num_timesteps=num_timesteps,
        initial_state=initial_state,
        seed=seed,
    )
    return states.numpy(), observations.numpy()


def _rmse(estimates: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimates - truth) ** 2)))


def _rmse_by_dim(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((estimates - truth) ** 2, axis=0))


def _rmse_per_timestep(estimates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((estimates - truth) ** 2, axis=1))


def _pf_means(particles: tf.Tensor, weights: tf.Tensor) -> np.ndarray:
    weights_t = weights[1:]
    particles_t = particles[1:]
    weighted = tf.einsum("tn,tnd->td", weights_t, particles_t)
    return weighted.numpy()


def _gaussian_log_likelihood(innovations: tf.Tensor, covs: tf.Tensor) -> float:
    log_two_pi = tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float32))
    num_steps = innovations.shape[0]

    def body(idx: tf.Tensor) -> tf.Tensor:
        innov = innovations[idx]
        cov = covs[idx]
        chol = tf.linalg.cholesky(cov)
        solved = tf.linalg.cholesky_solve(chol, innov[:, tf.newaxis])
        maha = tf.squeeze(tf.matmul(innov[tf.newaxis, :], solved), axis=0)
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        dim = tf.cast(tf.shape(innov)[0], tf.float32)
        return -0.5 * (maha + log_det + dim * log_two_pi)

    log_liks = tf.map_fn(body, tf.range(num_steps), fn_output_signature=tf.float32)
    return float(tf.reduce_sum(log_liks).numpy())


def _time_call(fn):
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    return result, end - start


def _run_particle_filter(
    model: NonlinearStateSpaceModel,
    observations: np.ndarray,
    truth: np.ndarray,
    num_particles: int,
    seed: int,
) -> FilterDiagnostics:
    tf.random.set_seed(seed)
    initial_particles = tf.random.normal(
        (num_particles, model.state_dim),
        dtype=tf.float32,
    )

    obs_tf = tf.convert_to_tensor(observations, dtype=tf.float32)

    def call():
        return particle_filter(
            model=model,
            observations=obs_tf,
            num_particles=num_particles,
            initial_particles=initial_particles,
        )

    result, runtime = _time_call(call)

    means = _pf_means(result.particles, result.weights)
    rmse = _rmse(means, truth)
    rmse_dim = _rmse_by_dim(means, truth)
    rmse_time = _rmse_per_timestep(means, truth)

    ess = result.effective_sample_sizes.numpy()
    ess_ratio = ess / float(num_particles)

    return FilterDiagnostics(
        rmse=rmse,
        rmse_by_dim=rmse_dim,
        rmse_per_timestep=rmse_time,
        log_likelihood=float(result.log_likelihood.numpy()),
        ess_min=float(np.min(ess)),
        ess_mean=float(np.mean(ess)),
        ess_ratio_min=float(np.min(ess_ratio)),
        ess_ratio_mean=float(np.mean(ess_ratio)),
        ess_ratio_sequence=ess_ratio,
        runtime_s=runtime,
        notes="Degeneracy visible when ESS ratio dips below resample threshold",
    )


def _run_differentiable_pf(
    model: NonlinearStateSpaceModel,
    observations: np.ndarray,
    truth: np.ndarray,
    num_particles: int,
    seed: int,
    mix_with_uniform: float,
    ot_epsilon: float,
    ot_num_iters: int,
) -> FilterDiagnostics:
    tf.random.set_seed(seed)
    initial_particles = tf.random.normal(
        (num_particles, model.state_dim),
        dtype=tf.float32,
    )

    obs_tf = tf.convert_to_tensor(observations, dtype=tf.float32)

    def call():
        return differentiable_particle_filter(
            model=model,
            observations=obs_tf,
            num_particles=num_particles,
            initial_particles=initial_particles,
            mix_with_uniform=mix_with_uniform,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
        )

    result, runtime = _time_call(call)

    means = _pf_means(result.particles, result.weights)
    rmse = _rmse(means, truth)
    rmse_dim = _rmse_by_dim(means, truth)
    rmse_time = _rmse_per_timestep(means, truth)

    ess = result.diagnostics["ess"].numpy()
    ess_ratio = ess / float(num_particles)

    return FilterDiagnostics(
        rmse=rmse,
        rmse_by_dim=rmse_dim,
        rmse_per_timestep=rmse_time,
        log_likelihood=float(result.log_likelihood.numpy()),
        ess_min=float(np.min(ess)),
        ess_mean=float(np.mean(ess)),
        ess_ratio_min=float(np.min(ess_ratio)),
        ess_ratio_mean=float(np.mean(ess_ratio)),
        ess_ratio_sequence=ess_ratio,
        runtime_s=runtime,
        notes="Soft OT mixing stabilizes ESS versus vanilla PF",
    )


def _run_ekf(
    model: NonlinearStateSpaceModel,
    observations: np.ndarray,
    truth: np.ndarray,
) -> FilterDiagnostics:
    initial_mean = tf.zeros((model.state_dim,), dtype=tf.float32)
    initial_cov = tf.eye(model.state_dim, dtype=tf.float32)
    obs_tf = tf.convert_to_tensor(observations, dtype=tf.float32)

    def call():
        return extended_kalman_filter(
            model=model,
            observations=obs_tf,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
        )

    result, runtime = _time_call(call)

    means = result.filtered_means.numpy()
    rmse = _rmse(means, truth)
    rmse_dim = _rmse_by_dim(means, truth)
    rmse_time = _rmse_per_timestep(means, truth)

    log_likelihood = _gaussian_log_likelihood(
        result.innovations,
        result.innovation_covs,
    )

    return FilterDiagnostics(
        rmse=rmse,
        rmse_by_dim=rmse_dim,
        rmse_per_timestep=rmse_time,
        log_likelihood=log_likelihood,
        ess_min=None,
        ess_mean=None,
        ess_ratio_min=None,
        ess_ratio_mean=None,
        ess_ratio_sequence=None,
        runtime_s=runtime,
        notes="EKF linearization accumulates error under strong sine nonlinearities",
    )


def _run_ukf(
    model: NonlinearStateSpaceModel,
    observations: np.ndarray,
    truth: np.ndarray,
) -> FilterDiagnostics:
    initial_mean = tf.zeros((model.state_dim,), dtype=tf.float32)
    initial_cov = tf.eye(model.state_dim, dtype=tf.float32)
    obs_tf = tf.convert_to_tensor(observations, dtype=tf.float32)

    def call():
        return unscented_kalman_filter(
            model=model,
            observations=obs_tf,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
        )

    result, runtime = _time_call(call)

    means = result.filtered_means.numpy()
    rmse = _rmse(means, truth)
    rmse_dim = _rmse_by_dim(means, truth)
    rmse_time = _rmse_per_timestep(means, truth)

    log_likelihood = _gaussian_log_likelihood(
        result.innovations,
        result.innovation_covs,
    )

    return FilterDiagnostics(
        rmse=rmse,
        rmse_by_dim=rmse_dim,
        rmse_per_timestep=rmse_time,
        log_likelihood=log_likelihood,
        ess_min=None,
        ess_mean=None,
        ess_ratio_min=None,
        ess_ratio_mean=None,
        ess_ratio_sequence=None,
        runtime_s=runtime,
        notes="Sigma points handle the observation nonlinearity better than EKF",
    )


def _aggregate(values: Iterable[FilterDiagnostics]) -> AggregateDiagnostics:
    runs = list(values)
    if not runs:
        raise ValueError("No diagnostics provided for aggregation")

    def _collect(attr: str, predicate=lambda x: True):
        collected = []
        for run in runs:
            value = getattr(run, attr)
            if not predicate(value):
                continue
            collected.append(value)
        if not collected:
            return None
        first = collected[0]
        if isinstance(first, np.ndarray):
            return np.stack(collected)
        return np.asarray(collected, dtype=np.float64)

    rmse = np.asarray([run.rmse for run in runs], dtype=np.float64)
    rmse_dim = _collect("rmse_by_dim")
    rmse_time = _collect("rmse_per_timestep")

    ll = _collect("log_likelihood", predicate=lambda v: v is not None)
    ess_min = _collect("ess_min", predicate=lambda v: v is not None)
    ess_mean = _collect("ess_mean", predicate=lambda v: v is not None)
    ess_ratio_min = _collect("ess_ratio_min", predicate=lambda v: v is not None)
    ess_ratio_mean = _collect("ess_ratio_mean", predicate=lambda v: v is not None)
    ess_ratio_seq = _collect("ess_ratio_sequence", predicate=lambda v: v is not None)
    runtime = np.asarray([run.runtime_s for run in runs], dtype=np.float64)

    def stats(arr: Optional[np.ndarray]) -> tuple[Optional[float], Optional[float]]:
        if arr is None:
            return None, None
        return float(np.mean(arr)), float(np.std(arr, ddof=0))

    ll_mean, ll_std = stats(ll)
    ess_min_mean, ess_min_std = stats(ess_min)
    ess_mean_mean, ess_mean_std = stats(ess_mean)
    ess_ratio_min_mean, ess_ratio_min_std = stats(ess_ratio_min)
    ess_ratio_mean_mean, ess_ratio_mean_std = stats(ess_ratio_mean)

    ess_ratio_seq_mean = None
    ess_ratio_seq_std = None
    if ess_ratio_seq is not None:
        ess_ratio_seq_mean = np.mean(ess_ratio_seq, axis=0).astype(float).tolist()
        ess_ratio_seq_std = np.std(ess_ratio_seq, axis=0, ddof=0).astype(float).tolist()

    return AggregateDiagnostics(
        rmse_mean=float(np.mean(rmse)),
        rmse_std=float(np.std(rmse, ddof=0)),
        rmse_by_dim_mean=rmse_dim.mean(axis=0).astype(float).tolist()
        if rmse_dim is not None
        else [],
        rmse_by_dim_std=rmse_dim.std(axis=0, ddof=0).astype(float).tolist()
        if rmse_dim is not None
        else [],
        rmse_per_timestep_mean=rmse_time.mean(axis=0).astype(float).tolist()
        if rmse_time is not None
        else [],
        rmse_per_timestep_std=rmse_time.std(axis=0, ddof=0).astype(float).tolist()
        if rmse_time is not None
        else [],
        log_likelihood_mean=ll_mean,
        log_likelihood_std=ll_std,
        ess_min_mean=ess_min_mean,
        ess_min_std=ess_min_std,
        ess_mean_mean=ess_mean_mean,
        ess_mean_std=ess_mean_std,
        ess_ratio_min_mean=ess_ratio_min_mean,
        ess_ratio_min_std=ess_ratio_min_std,
        ess_ratio_mean_mean=ess_ratio_mean_mean,
        ess_ratio_mean_std=ess_ratio_mean_std,
        ess_ratio_sequence_mean=ess_ratio_seq_mean,
        ess_ratio_sequence_std=ess_ratio_seq_std,
        runtime_mean=float(np.mean(runtime)),
        runtime_std=float(np.std(runtime, ddof=0)),
        notes=runs[0].notes,
        per_seed=runs,
    )


def _summarize(results: Dict[str, AggregateDiagnostics]) -> dict:
    summary = {}
    for name, diag in results.items():
        summary[name] = {
            "rmse_mean": diag.rmse_mean,
            "rmse_std": diag.rmse_std,
            "rmse_by_dim_mean": diag.rmse_by_dim_mean,
            "rmse_by_dim_std": diag.rmse_by_dim_std,
            "rmse_per_timestep_mean": diag.rmse_per_timestep_mean,
            "rmse_per_timestep_std": diag.rmse_per_timestep_std,
            "log_likelihood_mean": diag.log_likelihood_mean,
            "log_likelihood_std": diag.log_likelihood_std,
            "ess_min_mean": diag.ess_min_mean,
            "ess_min_std": diag.ess_min_std,
            "ess_mean_mean": diag.ess_mean_mean,
            "ess_mean_std": diag.ess_mean_std,
            "ess_ratio_min_mean": diag.ess_ratio_min_mean,
            "ess_ratio_min_std": diag.ess_ratio_min_std,
            "ess_ratio_mean_mean": diag.ess_ratio_mean_mean,
            "ess_ratio_mean_std": diag.ess_ratio_mean_std,
            "ess_ratio_sequence_mean": diag.ess_ratio_sequence_mean,
            "ess_ratio_sequence_std": diag.ess_ratio_sequence_std,
            "runtime_mean": diag.runtime_mean,
            "runtime_std": diag.runtime_std,
            "notes": diag.notes,
        }
    return summary


def _save_json(path: Optional[Path], payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _plot_rmse(
    timestep: np.ndarray,
    results: Dict[str, AggregateDiagnostics],
    path: Optional[Path],
) -> None:
    if path is None:
        return

    plt.figure(figsize=(7.0, 4.2))
    for name, diag in results.items():
        if not diag.rmse_per_timestep_mean:
            continue
        mean = np.array(diag.rmse_per_timestep_mean)
        std = np.array(diag.rmse_per_timestep_std)
        plt.plot(timestep, mean, label=name, linewidth=1.6)
        if np.any(std > 0):
            plt.fill_between(
                timestep,
                mean - std,
                mean + std,
                alpha=0.15,
            )
    plt.xlabel("Time step")
    plt.ylabel("RMSE")
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_ess(
    timestep: np.ndarray,
    results: Dict[str, AggregateDiagnostics],
    path: Optional[Path],
) -> None:
    if path is None:
        return

    plt.figure(figsize=(7.0, 4.0))
    for name, diag in results.items():
        if not diag.ess_ratio_sequence_mean:
            continue
        mean = np.array(diag.ess_ratio_sequence_mean)
        std = np.array(diag.ess_ratio_sequence_std)
        plt.plot(timestep, mean, label=name, linewidth=1.6)
        if np.any(std > 0):
            plt.fill_between(
                timestep,
                np.clip(mean - std, 0.0, 1.0),
                np.clip(mean + std, 0.999, 1.5),
                alpha=0.15,
            )
    plt.xlabel("Time step")
    plt.ylabel("ESS / N")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def evaluate_filters(
    model: NonlinearStateSpaceModel,
    num_steps: int,
    base_seed: int,
    num_seeds: int,
    num_particles: int,
    mix_with_uniform: float,
    ot_epsilon: float,
    ot_num_iters: int,
) -> Dict[str, AggregateDiagnostics]:
    results: Dict[str, list[FilterDiagnostics]] = {
        "ParticleFilter": [],
        "DifferentiablePF": [],
        "EKF": [],
        "UKF": [],
    }

    for i in range(num_seeds):
        sim_seed = base_seed + i
        truth, observations = _simulate_sequence(
            model,
            num_timesteps=num_steps,
            seed=sim_seed,
        )

        diag_pf = _run_particle_filter(
            model,
            observations,
            truth,
            num_particles=num_particles,
            seed=sim_seed,
        )
        diag_dpf = _run_differentiable_pf(
            model,
            observations,
            truth,
            num_particles=num_particles,
            seed=sim_seed,
            mix_with_uniform=mix_with_uniform,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
        )
        diag_ekf = _run_ekf(model, observations, truth)
        diag_ukf = _run_ukf(model, observations, truth)

        results["ParticleFilter"].append(diag_pf)
        results["DifferentiablePF"].append(diag_dpf)
        results["EKF"].append(diag_ekf)
        results["UKF"].append(diag_ukf)

    return {name: _aggregate(diags) for name, diags in results.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nonlinear filtering algorithms on the benchmark SSM.",
    )
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-particles", type=int, default=192)
    parser.add_argument("--mix-with-uniform", type=float, default=0.15)
    parser.add_argument("--ot-epsilon", type=float, default=0.25)
    parser.add_argument("--ot-num-iters", type=int, default=25)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--rmse-figure", type=Path, default=None)
    parser.add_argument("--ess-figure", type=Path, default=None)
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    model = _build_nonlinear_model()
    agg_results = evaluate_filters(
        model=model,
        num_steps=args.num_steps,
        base_seed=args.seed,
        num_seeds=args.num_seeds,
        num_particles=args.num_particles,
        mix_with_uniform=args.mix_with_uniform,
        ot_epsilon=args.ot_epsilon,
        ot_num_iters=args.ot_num_iters,
    )

    summary_payload = _summarize(agg_results)
    print(json.dumps(summary_payload, indent=2))

    _save_json(args.output_json, summary_payload)

    timestep = np.arange(args.num_steps)
    _plot_rmse(timestep, agg_results, args.rmse_figure)
    _plot_ess(timestep, agg_results, args.ess_figure)


if __name__ == "__main__":
    main()
