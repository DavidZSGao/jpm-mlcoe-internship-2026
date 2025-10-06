"""Tests for LinearGaussianSSM and Kalman filter implementations."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from mlcoe_q2.datasets import LinearGaussianSSM
from mlcoe_q2.filters import kalman_filter


def build_lgssm() -> LinearGaussianSSM:
    transition_matrix = tf.constant([[0.7, 0.2], [0.0, 0.9]], dtype=tf.float32)
    observation_matrix = tf.constant([[1.0, 0.0], [0.5, 1.0]], dtype=tf.float32)
    transition_cov = tf.constant([[0.1, 0.02], [0.02, 0.1]], dtype=tf.float32)
    observation_cov = tf.constant([[0.2, 0.01], [0.01, 0.3]], dtype=tf.float32)
    return LinearGaussianSSM(
        transition_matrix=transition_matrix,
        observation_matrix=observation_matrix,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
    )


def zero_noise_sampler(shape: tuple[int, ...], seed: int | None) -> tf.Tensor:
    del seed
    return tf.zeros(shape, dtype=tf.float32)


def numpy_kalman(model: LinearGaussianSSM, observations: np.ndarray) -> dict[str, np.ndarray]:
    A = model.transition_matrix.numpy()
    C = model.observation_matrix.numpy()
    Q = model.transition_cov.numpy()
    R = model.observation_cov.numpy()

    state_dim = A.shape[0]
    identity = np.eye(state_dim, dtype=np.float32)

    means = []
    covs = []
    innovations = []
    innovation_covs = []
    gains = []

    mean_pred = np.zeros(state_dim, dtype=np.float32)
    cov_pred = np.eye(state_dim, dtype=np.float32)

    for obs in observations:
        mean_pred = A @ mean_pred
        cov_pred = A @ cov_pred @ A.T + Q

        expected_obs = C @ mean_pred
        innovation = obs - expected_obs
        innovation_cov = C @ cov_pred @ C.T + R

        gain = cov_pred @ C.T @ np.linalg.inv(innovation_cov)

        mean_filt = mean_pred + gain @ innovation
        joseph_factor = identity - gain @ C
        cov_filt = (
            joseph_factor @ cov_pred @ joseph_factor.T
            + gain @ R @ gain.T
        )

        means.append(mean_filt)
        covs.append(cov_filt)
        innovations.append(innovation)
        innovation_covs.append(innovation_cov)
        gains.append(gain)

        mean_pred = mean_filt
        cov_pred = cov_filt

    return {
        "filtered_means": np.stack(means, axis=0),
        "filtered_covs": np.stack(covs, axis=0),
        "innovations": np.stack(innovations, axis=0),
        "innovation_covs": np.stack(innovation_covs, axis=0),
        "gains": np.stack(gains, axis=0),
    }


def test_simulation_shapes() -> None:
    model = build_lgssm()
    initial_state = tf.constant([0.5, -0.2], dtype=tf.float32)
    num_timesteps = 5

    states, observations = model.simulate(
        num_timesteps=num_timesteps,
        initial_state=initial_state,
        process_noise_sampler=zero_noise_sampler,
        observation_noise_sampler=zero_noise_sampler,
        seed=123,
    )

    assert states.shape == (num_timesteps, model.state_dim)
    assert observations.shape == (num_timesteps, model.observation_dim)

    tf.debugging.assert_near(states[0], tf.linalg.matvec(model.transition_matrix, initial_state))
    tf.debugging.assert_near(
        observations[0],
        tf.linalg.matvec(model.observation_matrix, states[0]),
    )


def test_kalman_filter_matches_numpy_reference() -> None:
    model = build_lgssm()
    observations = tf.constant(
        [
            [1.0, 0.4],
            [0.7, 0.6],
            [0.2, -0.1],
            [0.1, 0.0],
        ],
        dtype=tf.float32,
    )

    result = kalman_filter(
        model=model,
        observations=observations,
        initial_mean=tf.zeros(model.state_dim, dtype=tf.float32),
        initial_cov=tf.eye(model.state_dim, dtype=tf.float32),
    )

    reference = numpy_kalman(model, observations.numpy())

    for key in reference:
        actual = getattr(result, key).numpy()
        np.testing.assert_allclose(actual, reference[key], rtol=1e-5, atol=1e-6)
