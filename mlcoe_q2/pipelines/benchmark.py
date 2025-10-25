"""Benchmark routines for Question 2 filtering and flow methods."""

from __future__ import annotations

import json
import math
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.pipelines.flows import FlowBenchmarkResult, benchmark_flow
from mlcoe_q2.models.filters import (
    differentiable_particle_filter,
    extended_kalman_filter,
    particle_filter,
    particle_flow_particle_filter,
    unscented_kalman_filter,
)
from mlcoe_q2.models.filters.pfpf import ParticleFlowParticleFilterResult
from mlcoe_q2.models.flows import (
    ExactDaumHuangFlow,
    KernelEmbeddedFlow,
    LocalExactDaumHuangFlow,
    StochasticParticleFlow,
)

FlowFactory = Callable[[], Callable[..., FlowBenchmarkResult]]


@dataclass
class FilterBenchmark:
    """Summary statistics for a filtering method."""

    runtime_s: float
    peak_memory_kb: float
    log_likelihood: float
    ess_mean: Optional[float]
    aux_stats: Optional[dict[str, float]] = None


@dataclass
class FlowSequenceBenchmark:
    """Aggregate diagnostics when applying a flow across time steps."""

    total_runtime_s: float
    peak_memory_kb: float
    mean_particle_movement: float
    mean_residual_before: float
    mean_residual_after: float
    mean_log_jacobian: Optional[float] = None
    mean_kernel_condition: Optional[float] = None
    mean_movement_spread: Optional[float] = None


@dataclass
class BenchmarkSuiteResult:
    """Collection of benchmark results for Question 2 methods."""

    filter_results: dict[str, FilterBenchmark]
    flow_results: dict[str, FlowSequenceBenchmark]
    pfpf_results: dict[str, FilterBenchmark]
    def to_json(self) -> str:
        return json.dumps(
            {
                "filter_results": {
                    name: vars(result)
                    for name, result in self.filter_results.items()
                },
                "flow_results": {
                    name: vars(result)
                    for name, result in self.flow_results.items()
                },
                "pfpf_results": {
                    name: vars(result)
                    for name, result in self.pfpf_results.items()
                },
            },
            indent=2,
        )


def _build_nonlinear_model() -> NonlinearStateSpaceModel:
    transition_cov = tf.constant([[0.05, 0.0], [0.0, 0.03]], dtype=tf.float32)
    observation_cov = tf.constant([[0.2]], dtype=tf.float32)

    def transition_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        x0 = state[0]
        x1 = state[1]
        new_x0 = 0.85 * x0 + 0.2 * tf.math.sin(x1)
        new_x1 = 0.9 * x1 + 0.15 * tf.math.tanh(x0)
        return tf.stack([new_x0, new_x1])

    def observation_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        return tf.stack([0.6 * tf.math.sin(state[0]) + 0.1 * state[1]])

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
) -> tuple[tf.Tensor, tf.Tensor]:
    tf.random.set_seed(seed)
    initial_state = tf.constant([0.2, -0.1], dtype=tf.float32)
    states, observations = model.simulate(
        num_timesteps=num_timesteps,
        initial_state=initial_state,
        seed=seed,
    )
    return states, observations


def _time_and_profile(callable_fn: Callable[[], tf.Tensor]) -> tuple[tf.Tensor, float, float]:
    tracemalloc.start()
    start = time.perf_counter()
    result = callable_fn()
    end = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = end - start
    peak_kb = peak_mem / 1024.0
    return result, runtime, peak_kb


def _propagate_particles(
    model: NonlinearStateSpaceModel,
    particles: tf.Tensor,
    control: Optional[tf.Tensor],
    seed: int,
) -> tf.Tensor:
    tf.random.set_seed(seed)
    num_particles = tf.shape(particles)[0]
    noise = tf.random.normal((num_particles, model.state_dim), dtype=tf.float32)
    cov_chol = tf.linalg.cholesky(model.process_noise_cov)
    noise = noise @ tf.transpose(cov_chol)

    def transition_fn(particle: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(
            model.transition_fn(particle, control),
            dtype=tf.float32,
        )

    transitioned = tf.map_fn(
        transition_fn,
        particles,
        fn_output_signature=tf.float32,
    )
    return transitioned + noise


def _initial_particles(num_particles: int, state_dim: int, seed: int) -> tf.Tensor:
    tf.random.set_seed(seed)
    return tf.random.normal((num_particles, state_dim), dtype=tf.float32)


def _benchmark_particle_filter(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
    seed: int,
) -> FilterBenchmark:
    num_particles = 256
    initial_particles = _initial_particles(num_particles, model.state_dim, seed)

    result, runtime, peak = _time_and_profile(
        lambda: particle_filter(
            model=model,
            observations=observations,
            num_particles=num_particles,
            initial_particles=initial_particles,
        )
    )

    ess_mean = tf.reduce_mean(result.effective_sample_sizes).numpy()
    return FilterBenchmark(
        runtime_s=runtime,
        peak_memory_kb=peak,
        log_likelihood=float(result.log_likelihood.numpy()),
        ess_mean=float(ess_mean),
    )


def _benchmark_differentiable_pf(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
    seed: int,
) -> FilterBenchmark:
    num_particles = 256
    initial_particles = _initial_particles(num_particles, model.state_dim, seed)

    result, runtime, peak = _time_and_profile(
        lambda: differentiable_particle_filter(
            model=model,
            observations=observations,
            num_particles=num_particles,
            initial_particles=initial_particles,
            mix_with_uniform=0.15,
            ot_epsilon=0.25,
            ot_num_iters=25,
        )
    )

    ess_mean = tf.reduce_mean(result.diagnostics["ess"]).numpy()
    return FilterBenchmark(
        runtime_s=runtime,
        peak_memory_kb=peak,
        log_likelihood=float(result.log_likelihood.numpy()),
        ess_mean=float(ess_mean),
    )


def _benchmark_pfpf(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
    seed: int,
    flow_factory,
) -> FilterBenchmark:
    num_particles = 256
    initial_particles = _initial_particles(num_particles, model.state_dim, seed)

    def run() -> ParticleFlowParticleFilterResult:
        flow = flow_factory()
        return particle_flow_particle_filter(
            model=model,
            observations=observations,
            flow=flow,
            num_particles=num_particles,
            initial_particles=initial_particles,
            resample_threshold=0.5,
        )

    result, runtime, peak = _time_and_profile(run)

    ess_mean = tf.reduce_mean(result.effective_sample_sizes).numpy()
    mean_log_jac = tf.reduce_mean(result.flow_log_jacobians).numpy()
    return FilterBenchmark(
        runtime_s=runtime,
        peak_memory_kb=peak,
        log_likelihood=float(result.log_likelihood.numpy()),
        ess_mean=float(ess_mean),
        aux_stats={
            "mean_flow_log_jacobian": float(mean_log_jac),
        },
    )


def _gaussian_log_likelihood(innovations: tf.Tensor, covs: tf.Tensor) -> float:
    log_two_pi = tf.math.log(tf.constant(2.0 * math.pi, dtype=tf.float32))

    def body(idx: tf.Tensor) -> tf.Tensor:
        innov = innovations[idx]
        cov = covs[idx]
        chol = tf.linalg.cholesky(cov)
        solved = tf.linalg.cholesky_solve(chol, innov[:, tf.newaxis])
        maha = tf.squeeze(tf.matmul(innov[tf.newaxis, :], solved), axis=0)
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        dim = tf.cast(tf.shape(innov)[0], tf.float32)
        return -0.5 * (maha + log_det + dim * log_two_pi)

    num_steps = tf.shape(innovations)[0]
    log_liks = tf.map_fn(body, tf.range(num_steps), fn_output_signature=tf.float32)
    return float(tf.reduce_sum(log_liks).numpy())


def _benchmark_ekf(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
) -> FilterBenchmark:
    initial_mean = tf.zeros((model.state_dim,), dtype=tf.float32)
    initial_cov = tf.eye(model.state_dim, dtype=tf.float32)

    result, runtime, peak = _time_and_profile(
        lambda: extended_kalman_filter(
            model=model,
            observations=observations,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
        )
    )

    log_likelihood = _gaussian_log_likelihood(
        innovations=result.innovations,
        covs=result.innovation_covs,
    )
    return FilterBenchmark(
        runtime_s=runtime,
        peak_memory_kb=peak,
        log_likelihood=log_likelihood,
        ess_mean=None,
    )


def _benchmark_ukf(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
) -> FilterBenchmark:
    initial_mean = tf.zeros((model.state_dim,), dtype=tf.float32)
    initial_cov = tf.eye(model.state_dim, dtype=tf.float32)

    result, runtime, peak = _time_and_profile(
        lambda: unscented_kalman_filter(
            model=model,
            observations=observations,
            initial_mean=initial_mean,
            initial_cov=initial_cov,
        )
    )

    log_likelihood = _gaussian_log_likelihood(
        innovations=result.innovations,
        covs=result.innovation_covs,
    )
    return FilterBenchmark(
        runtime_s=runtime,
        peak_memory_kb=peak,
        log_likelihood=log_likelihood,
        ess_mean=None,
    )


def _flow_sequence_benchmark(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
    flow,
    seed: int,
) -> FlowSequenceBenchmark:
    num_particles = 256
    particles = _initial_particles(num_particles, model.state_dim, seed)
    weights = tf.fill((num_particles,), 1.0 / float(num_particles))

    total_runtime = 0.0
    peak_memory = 0.0
    movement = []
    residual_before = []
    residual_after = []
    log_jacobian_norms: list[float] = []
    kernel_conditions: list[float] = []
    movement_spreads: list[float] = []

    for t in range(observations.shape[0]):
        particles = _propagate_particles(
            model,
            particles,
            control=None,
            seed=seed + t,
        )
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

        total_runtime += bm.runtime_s
        peak_memory = max(peak_memory, bm.peak_memory_kb)

        diagnostics = bm.diagnostics
        propagated_particles = diagnostics.get("propagated_particles")
        propagated_weights = diagnostics.get("propagated_weights")

        if propagated_particles is not None:
            particles = tf.convert_to_tensor(propagated_particles[-1])
        if propagated_weights is not None:
            weights = tf.convert_to_tensor(propagated_weights[-1])

        movement.append(diagnostics.get("mean_particle_movement", [0.0])[-1])
        residual_before.append(diagnostics.get("mean_residual_before", [0.0])[-1])
        residual_after.append(diagnostics.get("mean_residual_after", [0.0])[-1])

        log_jac = diagnostics.get("log_jacobian_norm")
        if log_jac:
            log_jacobian_norms.append(log_jac[-1])

        kernel_cond = diagnostics.get("kernel_condition")
        if kernel_cond:
            kernel_conditions.append(kernel_cond[-1])

        movement_dist = diagnostics.get("particle_movement_distribution")
        if movement_dist:
            step_movement = tf.convert_to_tensor(
                movement_dist[-1], dtype=tf.float32
            )
            movement_spreads.append(
                float(tf.math.reduce_std(step_movement).numpy())
            )

    return FlowSequenceBenchmark(
        total_runtime_s=total_runtime,
        peak_memory_kb=peak_memory,
        mean_particle_movement=float(tf.reduce_mean(movement).numpy()),
        mean_residual_before=float(tf.reduce_mean(residual_before).numpy()),
        mean_residual_after=float(tf.reduce_mean(residual_after).numpy()),
        mean_log_jacobian=(
            float(
                tf.reduce_mean(tf.convert_to_tensor(log_jacobian_norms, dtype=tf.float32)).numpy()
            )
            if log_jacobian_norms
            else None
        ),
        mean_kernel_condition=(
            float(
                tf.reduce_mean(tf.convert_to_tensor(kernel_conditions, dtype=tf.float32)).numpy()
            )
            if kernel_conditions
            else None
        ),
        mean_movement_spread=(
            float(
                tf.reduce_mean(tf.convert_to_tensor(movement_spreads, dtype=tf.float32)).numpy()
            )
            if movement_spreads
            else None
        ),
    )


def run_benchmark_suite(
    num_timesteps: int = 15,
    seed: int = 0,
) -> BenchmarkSuiteResult:
    model = _build_nonlinear_model()
    _, observations = _simulate_sequence(model, num_timesteps, seed)

    filters = {
        "PF": _benchmark_particle_filter(model, observations, seed),
        "DifferentiablePF": _benchmark_differentiable_pf(model, observations, seed),
        "EKF": _benchmark_ekf(model, observations),
        "UKF": _benchmark_ukf(model, observations),
    }

    flows = {
        "EDH": _flow_sequence_benchmark(
            model,
            observations,
            flow=ExactDaumHuangFlow(step_size=1.0, num_steps=4),
            seed=seed,
        ),
        "LEDH": _flow_sequence_benchmark(
            model,
            observations,
            flow=LocalExactDaumHuangFlow(step_size=0.8, num_steps=3),
            seed=seed,
        ),
        "KernelScalar": _flow_sequence_benchmark(
            model,
            observations,
            flow=KernelEmbeddedFlow(
                kernel_type="scalar",
                bandwidth=1.2,
                step_size=0.5,
                num_steps=3,
            ),
            seed=seed,
        ),
        "KernelDiagonal": _flow_sequence_benchmark(
            model,
            observations,
            flow=KernelEmbeddedFlow(
                kernel_type="diagonal",
                bandwidth=1.2,
                step_size=0.5,
                num_steps=3,
            ),
            seed=seed,
        ),
        "KernelMatrix": _flow_sequence_benchmark(
            model,
            observations,
            flow=KernelEmbeddedFlow(
                kernel_type="matrix",
                bandwidth=1.2,
                step_size=0.5,
                num_steps=3,
            ),
            seed=seed,
        ),
        "Stochastic": _flow_sequence_benchmark(
            model,
            observations,
            flow=StochasticParticleFlow(
                step_size=0.8,
                num_steps=6,
                diffusion=0.08,
            ),
            seed=seed,
        ),
    }

    pfpf = {
        "PF_PF_EDH": _benchmark_pfpf(
            model,
            observations,
            seed,
            lambda: ExactDaumHuangFlow(step_size=1.0, num_steps=4),
        ),
        "PF_PF_LEDH": _benchmark_pfpf(
            model,
            observations,
            seed,
            lambda: LocalExactDaumHuangFlow(step_size=0.8, num_steps=3),
        ),
        "PF_PF_KernelScalar": _benchmark_pfpf(
            model,
            observations,
            seed,
            lambda: KernelEmbeddedFlow(
                kernel_type="scalar",
                bandwidth=1.2,
                step_size=0.5,
                num_steps=3,
            ),
        ),
        "PF_PF_KernelMatrix": _benchmark_pfpf(
            model,
            observations,
            seed,
            lambda: KernelEmbeddedFlow(
                kernel_type="matrix",
                bandwidth=1.2,
                step_size=0.5,
                num_steps=3,
            ),
        ),
    }

    return BenchmarkSuiteResult(filter_results=filters, flow_results=flows, pfpf_results=pfpf)


if __name__ == "__main__":
    suite = run_benchmark_suite()
    print(suite.to_json())
