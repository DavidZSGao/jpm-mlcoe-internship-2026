"""Tests for deterministic particle flow implementations."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from mlcoe_q2.data import NonlinearStateSpaceModel
from mlcoe_q2.pipelines import FlowBenchmarkResult, benchmark_flow
from mlcoe_q2.models.flows import (
    ExactDaumHuangFlow,
    KernelEmbeddedFlow,
    LocalExactDaumHuangFlow,
)


def _build_nonlinear_model() -> NonlinearStateSpaceModel:
    transition_cov = tf.constant([[0.05, 0.0], [0.0, 0.03]], dtype=tf.float32)
    observation_cov = tf.constant([[0.1]], dtype=tf.float32)

    def transition_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        x0 = state[0]
        x1 = state[1]
        new_x0 = 0.9 * x0 + 0.2 * tf.math.sin(x1)
        new_x1 = 0.95 * x1 + 0.1 * tf.math.tanh(x0)
        return tf.stack([new_x0, new_x1])

    def observation_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        return tf.stack([tf.math.sin(state[0]) + 0.1 * state[1]])

    return NonlinearStateSpaceModel(
        state_dim=2,
        observation_dim=1,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        process_noise_cov=transition_cov,
        observation_noise_cov=observation_cov,
    )


def _initial_particles(num_particles: int, state_dim: int) -> tf.Tensor:
    return tf.random.normal((num_particles, state_dim), dtype=tf.float32)


def _mean_residual_norm(
    model: NonlinearStateSpaceModel,
    particles: tf.Tensor,
    observation: tf.Tensor,
) -> tf.Tensor:
    obs_pred = tf.vectorized_map(
        lambda p: tf.convert_to_tensor(model.observation_fn(p, None), dtype=tf.float32),
        particles,
    )
    residuals = obs_pred - observation
    return tf.reduce_mean(tf.linalg.norm(residuals, axis=-1))


def _movement_magnitude(before: tf.Tensor, after: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.linalg.norm(after - before, axis=1))


def _assert_flow_behaviour(result, particles_before, model, observation) -> None:
    np.testing.assert_array_equal(
        result.propagated_particles.shape, particles_before.shape
    )
    np.testing.assert_array_equal(
        result.propagated_weights.shape,
        (particles_before.shape[0],),
    )
    assert tf.reduce_all(tf.math.is_finite(result.propagated_particles))
    assert tf.reduce_all(tf.math.is_finite(result.propagated_weights))

    movement = _movement_magnitude(particles_before, result.propagated_particles)
    assert movement.numpy() > 0.0

    mean_residual_after = _mean_residual_norm(
        model,
        result.propagated_particles,
        observation,
    )
    mean_residual_before = _mean_residual_norm(model, particles_before, observation)
    assert mean_residual_after.numpy() <= mean_residual_before.numpy() + 0.25

    for diag_value in result.diagnostics.values():
        assert tf.reduce_all(tf.math.is_finite(diag_value))


def test_exact_daum_huang_flow_updates_particles() -> None:
    tf.random.set_seed(0)
    model = _build_nonlinear_model()
    particles = _initial_particles(num_particles=64, state_dim=model.state_dim)
    weights = tf.ones((particles.shape[0],), dtype=tf.float32)
    observation = tf.constant([0.25], dtype=tf.float32)

    flow = ExactDaumHuangFlow(step_size=1.0, num_steps=4)
    result = flow(
        model=model,
        particles=particles,
        weights=weights,
        observation=observation,
    )

    assert "grad_norm_mean" in result.diagnostics
    assert "residual_norm_mean" in result.diagnostics
    _assert_flow_behaviour(result, particles, model, observation)


def test_local_exact_daum_huang_flow_updates_particles() -> None:
    tf.random.set_seed(1)
    model = _build_nonlinear_model()
    particles = _initial_particles(num_particles=64, state_dim=model.state_dim)
    weights = tf.ones((particles.shape[0],), dtype=tf.float32)
    observation = tf.constant([0.15], dtype=tf.float32)

    flow = LocalExactDaumHuangFlow(step_size=0.8, num_steps=3)
    result = flow(
        model=model,
        particles=particles,
        weights=weights,
        observation=observation,
    )

    assert "delta_norm_mean" in result.diagnostics
    assert "innovation_cond_mean" in result.diagnostics
    _assert_flow_behaviour(result, particles, model, observation)


def test_kernel_embedded_flow_updates_particles() -> None:
    tf.random.set_seed(2)
    model = _build_nonlinear_model()
    particles = _initial_particles(num_particles=64, state_dim=model.state_dim)
    weights = tf.ones((particles.shape[0],), dtype=tf.float32)
    observation = tf.constant([0.05], dtype=tf.float32)

    flow = KernelEmbeddedFlow(
        kernel_type="scalar",
        bandwidth=1.2,
        step_size=0.5,
        num_steps=3,
    )
    result = flow(
        model=model,
        particles=particles,
        weights=weights,
        observation=observation,
    )

    assert "kernel_trace" in result.diagnostics
    assert "score_norm_mean" in result.diagnostics
    _assert_flow_behaviour(result, particles, model, observation)


def test_benchmark_flow_returns_metrics() -> None:
    tf.random.set_seed(3)
    model = _build_nonlinear_model()
    particles = _initial_particles(num_particles=32, state_dim=model.state_dim)
    weights = tf.ones((particles.shape[0],), dtype=tf.float32)
    observation = tf.constant([0.05], dtype=tf.float32)

    flow = ExactDaumHuangFlow(step_size=0.5, num_steps=2)
    result = benchmark_flow(
        model=model,
        flow=flow,
        particles=particles,
        weights=weights,
        observation=observation,
        warmup=0,
        num_repeats=1,
    )

    assert isinstance(result, FlowBenchmarkResult)
    assert result.runtime_s >= 0.0
    assert result.peak_memory_kb >= 0.0
    assert result.mean_particle_movement > 0.0
    assert "grad_norm_mean" in result.diagnostics
