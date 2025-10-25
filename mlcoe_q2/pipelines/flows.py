"""Utilities for benchmarking deterministic particle flow filters."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.models.flows.base import ParticleFlowResult

FlowCallable = Callable[
    [
        NonlinearStateSpaceModel,
        tf.Tensor,
        tf.Tensor,
        tf.Tensor,
        Optional[tf.Tensor],
    ],
    ParticleFlowResult,
]


@dataclass
class FlowBenchmarkResult:
    """Summary statistics from a particle flow benchmark run."""

    runtime_s: float
    peak_memory_kb: float
    mean_residual_before: float
    mean_residual_after: float
    mean_particle_movement: float
    diagnostics: dict[str, list[float]]


def benchmark_flow(
    model: NonlinearStateSpaceModel,
    flow: FlowCallable,
    particles: tf.Tensor,
    weights: tf.Tensor,
    observation: tf.Tensor,
    control: Optional[tf.Tensor] = None,
    *,
    warmup: int = 1,
    num_repeats: int = 3,
) -> FlowBenchmarkResult:
    """Benchmark a deterministic particle flow on a single observation.

    Args:
        model: Nonlinear state-space model used for flow propagation.
        flow: Callable implementing the particle flow interface.
        particles: Input particles of shape `[num_particles, state_dim]`.
        weights: Particle weights of shape `[num_particles]`.
        observation: Observation tensor of shape `[observation_dim]`.
        control: Optional control input tensor.
        warmup: Number of warmup evaluations before timing.
        num_repeats: Number of timed repetitions used to average runtime.

    Returns:
        `FlowBenchmarkResult` containing runtime, peak memory, and diagnostics.
    """

    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    observation = tf.convert_to_tensor(observation, dtype=tf.float32)
    control = (
        tf.convert_to_tensor(control, dtype=tf.float32)
        if control is not None
        else None
    )

    def run_flow() -> ParticleFlowResult:
        return flow(
            model=model,
            particles=particles,
            weights=weights,
            observation=observation,
            control=control,
        )

    # Warmup evaluations to stabilize any tracing overhead.
    for _ in range(max(warmup, 0)):
        _ = run_flow()

    tracemalloc.start()
    start = time.perf_counter()
    result = None
    for _ in range(max(num_repeats, 1)):
        result = run_flow()
    end = time.perf_counter()
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result is not None, "Flow callable did not produce a result"

    runtime_s = (end - start) / max(num_repeats, 1)
    peak_memory_kb = peak_memory_bytes / 1024.0

    mean_residual_before = _mean_residual_norm(model, particles, observation)
    mean_residual_after = _mean_residual_norm(
        model,
        result.propagated_particles,
        observation,
    )
    mean_particle_movement = _particle_movement(
        particles,
        result.propagated_particles,
    )

    diagnostics = {
        key: _to_list(value) for key, value in result.diagnostics.items()
    }
    diagnostics["propagated_particles"] = [
        tf.convert_to_tensor(
            result.propagated_particles,
            dtype=tf.float32,
        ).numpy(),
    ]
    diagnostics["propagated_weights"] = [
        tf.convert_to_tensor(
            result.propagated_weights,
            dtype=tf.float32,
        ).numpy(),
    ]
    diagnostics.setdefault("mean_residual_before", []).append(
        float(mean_residual_before.numpy())
    )
    diagnostics.setdefault("mean_residual_after", []).append(
        float(mean_residual_after.numpy())
    )
    diagnostics.setdefault("mean_particle_movement", []).append(
        float(mean_particle_movement.numpy())
    )
    diagnostics.setdefault("log_jacobian_norm", []).append(
        float(tf.reduce_mean(tf.abs(result.log_jacobians)).numpy())
    )

    movement_by_step = tf.linalg.norm(
        result.propagated_particles - particles,
        axis=1,
    )
    diagnostics.setdefault("particle_movement_distribution", []).append(
        movement_by_step.numpy().astype(float).tolist()
    )

    return FlowBenchmarkResult(
        runtime_s=runtime_s,
        peak_memory_kb=peak_memory_kb,
        mean_residual_before=float(mean_residual_before.numpy()),
        mean_residual_after=float(mean_residual_after.numpy()),
        mean_particle_movement=float(mean_particle_movement.numpy()),
        diagnostics=diagnostics,
    )


def _mean_residual_norm(
    model: NonlinearStateSpaceModel,
    particles: tf.Tensor,
    observation: tf.Tensor,
) -> tf.Tensor:
    def obs_map(particle: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(
            model.observation_fn(particle, None), dtype=tf.float32
        )

    predicted = tf.vectorized_map(obs_map, particles)
    residuals = predicted - observation
    return tf.reduce_mean(tf.linalg.norm(residuals, axis=-1))


def _particle_movement(before: tf.Tensor, after: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.linalg.norm(after - before, axis=1))


def _to_list(tensor: tf.Tensor) -> list[float]:
    array = tf.convert_to_tensor(tensor, dtype=tf.float32).numpy()
    return array.astype(float).flatten().tolist()
