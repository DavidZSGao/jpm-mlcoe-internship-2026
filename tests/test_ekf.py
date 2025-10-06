"""Tests for the extended Kalman filter implementation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from mlcoe_q2.datasets import LinearGaussianSSM, NonlinearStateSpaceModel
from mlcoe_q2.filters import extended_kalman_filter, kalman_filter


def _build_linear_system() -> tuple[LinearGaussianSSM, NonlinearStateSpaceModel]:
    transition_matrix = tf.constant([[0.8, 0.1], [0.0, 0.9]], dtype=tf.float32)
    observation_matrix = tf.constant([[1.0, 0.0], [0.3, 1.0]], dtype=tf.float32)
    transition_cov = tf.constant([[0.05, 0.01], [0.01, 0.05]], dtype=tf.float32)
    observation_cov = tf.constant([[0.2, 0.0], [0.0, 0.25]], dtype=tf.float32)

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


def test_ekf_matches_kalman_with_linear_jacobians() -> None:
    linear_model, nonlinear_model = _build_linear_system()

    observations = tf.constant(
        [
            [0.9, 0.6],
            [0.5, 0.4],
            [0.7, 0.5],
            [0.2, 0.1],
        ],
        dtype=tf.float32,
    )

    initial_mean = tf.constant([0.0, 0.0], dtype=tf.float32)
    initial_cov = tf.eye(linear_model.state_dim, dtype=tf.float32)

    kf_result = kalman_filter(
        model=linear_model,
        observations=observations,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )

    transition_matrix = linear_model.transition_matrix
    observation_matrix = linear_model.observation_matrix

    def transition_jacobian(_: tf.Tensor, __: tf.Tensor | None) -> tf.Tensor:
        return transition_matrix

    def observation_jacobian(_: tf.Tensor, __: tf.Tensor | None) -> tf.Tensor:
        return observation_matrix

    ekf_result = extended_kalman_filter(
        model=nonlinear_model,
        observations=observations,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
        transition_jacobian_fn=transition_jacobian,
        observation_jacobian_fn=observation_jacobian,
    )

    for field in ["filtered_means", "filtered_covs", "innovations", "innovation_covs"]:
        np.testing.assert_allclose(
            getattr(ekf_result, field).numpy(),
            getattr(kf_result, field).numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


def test_ekf_autodiff_jacobians_match_manual() -> None:
    linear_model, nonlinear_model = _build_linear_system()

    observations = tf.constant(
        [
            [0.9, 0.6],
            [0.5, 0.4],
            [0.7, 0.5],
            [0.2, 0.1],
        ],
        dtype=tf.float32,
    )

    initial_mean = tf.constant([0.1, -0.2], dtype=tf.float32)
    initial_cov = tf.eye(linear_model.state_dim, dtype=tf.float32)

    kf_result = kalman_filter(
        model=linear_model,
        observations=observations,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )

    ekf_result = extended_kalman_filter(
        model=nonlinear_model,
        observations=observations,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )

    np.testing.assert_allclose(
        ekf_result.filtered_means.numpy(),
        kf_result.filtered_means.numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ekf_result.filtered_covs.numpy(),
        kf_result.filtered_covs.numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
