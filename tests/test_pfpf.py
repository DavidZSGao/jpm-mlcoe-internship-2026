
"""Tests for the particle-flow particle filter implementation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from mlcoe_q2.datasets import LinearGaussianSSM, NonlinearStateSpaceModel
from mlcoe_q2.filters import particle_filter, particle_flow_particle_filter
from mlcoe_q2.flows import ExactDaumHuangFlow
from mlcoe_q2.flows.base import ParticleFlowResult


class IdentityFlow:
    """Flow that leaves particles unchanged (for regression tests)."""

    def __call__(
        self,
        model: NonlinearStateSpaceModel,
        particles: tf.Tensor,
        weights: tf.Tensor,
        observation: tf.Tensor,
        control: tf.Tensor | None = None,
    ) -> ParticleFlowResult:
        del model, observation, control
        return ParticleFlowResult(
            propagated_particles=tf.convert_to_tensor(particles, dtype=tf.float32),
            propagated_weights=tf.convert_to_tensor(weights, dtype=tf.float32),
            log_jacobians=tf.zeros(tf.shape(weights), dtype=tf.float32),
            diagnostics={},
        )


def _build_linear_system() -> tuple[LinearGaussianSSM, NonlinearStateSpaceModel]:
    transition_matrix = tf.constant([[0.9, 0.1], [0.0, 0.95]], dtype=tf.float32)
    observation_matrix = tf.constant([[1.0, 0.0]], dtype=tf.float32)
    transition_cov = tf.constant(
        [[0.05, 0.0], [0.0, 0.03]], dtype=tf.float32
    )
    observation_cov = tf.constant([[0.2]], dtype=tf.float32)

    linear_model = LinearGaussianSSM(
        transition_matrix=transition_matrix,
        observation_matrix=observation_matrix,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
    )

    def transition_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        return tf.linalg.matvec(transition_matrix, state)

    def observation_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        return tf.linalg.matvec(observation_matrix, state)

    nonlinear_model = NonlinearStateSpaceModel(
        state_dim=linear_model.state_dim,
        observation_dim=linear_model.observation_dim,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        process_noise_cov=transition_cov,
        observation_noise_cov=observation_cov,
    )

    return linear_model, nonlinear_model


def _simulate_linear_sequence(
    model: LinearGaussianSSM,
    num_timesteps: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    tf.random.set_seed(seed)
    initial_state = tf.constant([0.1, -0.2], dtype=tf.float32)
    return model.simulate(num_timesteps=num_timesteps, initial_state=initial_state, seed=seed)


def _build_nonlinear_model() -> NonlinearStateSpaceModel:
    process_cov = tf.constant([[0.05, 0.0], [0.0, 0.03]], dtype=tf.float32)
    observation_cov = tf.constant([[0.2]], dtype=tf.float32)

    def transition_fn(state: tf.Tensor, control: tf.Tensor | None) -> tf.Tensor:
        del control
        x0, x1 = state[0], state[1]
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
        process_noise_cov=process_cov,
        observation_noise_cov=observation_cov,
    )


def test_pfpf_matches_standard_pf_with_identity_flow() -> None:
    linear_model, nonlinear_model = _build_linear_system()
    _, observations = _simulate_linear_sequence(linear_model, num_timesteps=4, seed=5)

    num_particles = 32
    tf.random.set_seed(12)
    initial_particles = tf.random.normal((num_particles, nonlinear_model.state_dim), dtype=tf.float32)

    tf.random.set_seed(99)
    pf_result = particle_filter(
        model=nonlinear_model,
        observations=observations,
        num_particles=num_particles,
        initial_particles=initial_particles,
        resample_threshold=0.0,
    )

    tf.random.set_seed(99)
    pfpf_result = particle_flow_particle_filter(
        model=nonlinear_model,
        observations=observations,
        flow=IdentityFlow(),
        num_particles=num_particles,
        initial_particles=initial_particles,
        resample_threshold=0.0,
    )

    np.testing.assert_allclose(
        pf_result.log_likelihood.numpy(),
        pfpf_result.log_likelihood.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        pf_result.weights[-1].numpy(),
        pfpf_result.weights[-1].numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_pfpf_with_edh_flow_produces_finite_log_jacobians() -> None:
    model = _build_nonlinear_model()
    tf.random.set_seed(7)
    states, observations = model.simulate(num_timesteps=4, initial_state=tf.constant([0.2, -0.1], dtype=tf.float32))

    num_particles = 32
    tf.random.set_seed(21)
    initial_particles = tf.random.normal((num_particles, model.state_dim), dtype=tf.float32)

    flow = ExactDaumHuangFlow(step_size=0.6, num_steps=3)
    result = particle_flow_particle_filter(
        model=model,
        observations=observations,
        flow=flow,
        num_particles=num_particles,
        initial_particles=initial_particles,
    )

    assert result.flow_log_jacobians.shape == (observations.shape[0], num_particles)
    assert tf.reduce_all(tf.math.is_finite(result.flow_log_jacobians))
    assert tf.reduce_any(tf.not_equal(result.flow_log_jacobians, 0.0))
    assert np.isfinite(result.log_likelihood.numpy())
    assert result.flow_diagnostics, "Diagnostics should be populated"

    estimates = tf.reduce_sum(
        result.weights[-1][:, tf.newaxis] * result.particles[-1], axis=0
    )
    assert tf.reduce_all(tf.math.is_finite(estimates))
