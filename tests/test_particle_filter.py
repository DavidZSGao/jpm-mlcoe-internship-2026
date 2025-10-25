"""Tests for the particle filter implementation."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import tensorflow as tf

from mlcoe_q2.data import LinearGaussianSSM, NonlinearStateSpaceModel
from mlcoe_q2.models.filters import kalman_filter, particle_filter


def _build_linear_ssm() -> Tuple[LinearGaussianSSM, NonlinearStateSpaceModel]:
    transition_matrix = tf.constant(
        [[0.9, 0.1], [0.0, 0.95]], dtype=tf.float32
    )
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


def _kalman_log_likelihood(
    innovations: tf.Tensor, covariances: tf.Tensor
) -> tf.Tensor:
    log_likelihood = tf.constant(0.0, dtype=tf.float32)
    obs_dim = innovations.shape[-1]
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float32)

    for t in range(innovations.shape[0]):
        innovation = tf.reshape(innovations[t], (obs_dim, 1))
        cov = tf.reshape(covariances[t], (obs_dim, obs_dim))
        log_det = tf.linalg.logdet(cov)
        solved = tf.linalg.solve(cov, innovation)
        quad = tf.matmul(innovation, solved, transpose_a=True)
        quad = tf.reshape(quad, ())
        log_likelihood = log_likelihood - 0.5 * (
            log_det + quad + obs_dim * tf.math.log(two_pi)
        )

    return log_likelihood


def test_particle_filter_matches_kalman_on_linear_model() -> None:
    linear_model, nonlinear_model = _build_linear_ssm()

    initial_state = tf.constant([0.0, 0.0], dtype=tf.float32)
    observations = tf.constant(
        [[0.2], [0.1], [-0.05], [0.15], [0.3]], dtype=tf.float32
    )

    kf_result = kalman_filter(
        model=linear_model,
        observations=observations,
        initial_mean=initial_state,
        initial_cov=tf.eye(linear_model.state_dim, dtype=tf.float32),
    )

    # Use a moderate particle count to keep runtime manageable while still
    # approximating the Kalman posterior with low variance.
    num_particles = 1024
    tf.random.set_seed(0)
    initial_particles = tf.random.normal(
        (num_particles, nonlinear_model.state_dim), dtype=tf.float32
    )

    pf_result = particle_filter(
        model=nonlinear_model,
        observations=observations,
        num_particles=num_particles,
        initial_particles=initial_particles,
    )

    pf_mean = tf.reduce_sum(
        pf_result.weights[-1][:, tf.newaxis] * pf_result.particles[-1], axis=0
    )

    tf.debugging.assert_near(
        pf_mean,
        kf_result.filtered_means[-1],
        atol=0.2,
        rtol=0.15,
    )

    log_likelihood_kf = _kalman_log_likelihood(
        kf_result.innovations,
        kf_result.innovation_covs,
    )

    tf.debugging.assert_less(
        tf.abs(pf_result.log_likelihood - log_likelihood_kf),
        tf.constant(5.0, dtype=tf.float32),
    )


def test_particle_filter_effective_sample_size_triggers_resample() -> None:
    _, nonlinear_model = _build_linear_ssm()

    observations = tf.constant(
        [[0.2], [0.1], [0.05], [-0.02], [0.03]], dtype=tf.float32
    )

    num_particles = 256
    initial_particles = tf.zeros(
        (num_particles, nonlinear_model.state_dim), dtype=tf.float32
    )

    pf_result = particle_filter(
        model=nonlinear_model,
        observations=observations,
        num_particles=num_particles,
        initial_particles=initial_particles,
        resample_threshold=0.7,
    )

    ess = pf_result.effective_sample_sizes.numpy()
    assert np.any(ess < 0.7 * num_particles)

    ancestor_indices = pf_result.ancestor_indices.numpy()
    assert np.any(ancestor_indices[:-1] != ancestor_indices[1:])
