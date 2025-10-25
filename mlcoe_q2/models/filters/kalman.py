"""TensorFlow implementation of the Kalman filter for LGSSMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from mlcoe_q2.data.lgssm import LinearGaussianSSM


@dataclass
class KalmanFilterResult:
    """Container for Kalman filtering outputs."""

    filtered_means: tf.Tensor
    filtered_covs: tf.Tensor
    innovations: tf.Tensor
    innovation_covs: tf.Tensor
    gains: tf.Tensor


def kalman_filter(
    model: LinearGaussianSSM,
    observations: tf.Tensor,
    initial_mean: tf.Tensor,
    initial_cov: tf.Tensor,
    controls: Optional[tf.Tensor] = None,
) -> KalmanFilterResult:
    """Run the Kalman filter on a sequence of observations."""

    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    initial_mean = tf.convert_to_tensor(initial_mean, dtype=tf.float32)
    initial_cov = tf.convert_to_tensor(initial_cov, dtype=tf.float32)

    if observations.shape[-1] != model.observation_dim:
        raise ValueError("Observation dimension mismatch")
    if initial_mean.shape[-1] != model.state_dim:
        raise ValueError("Initial mean dimension mismatch")
    if initial_cov.shape[-1] != model.state_dim:
        raise ValueError("Initial covariance dimension mismatch")
    if controls is not None:
        controls = tf.convert_to_tensor(controls, dtype=tf.float32)
        expected_shape = (
            observations.shape[0],
            model.control_matrix.shape[-1],
        )
        if controls.shape != expected_shape:
            raise ValueError(
                "controls must have shape "
                f"{expected_shape}, got {controls.shape}"
            )

    num_timesteps = observations.shape[0]

    filtered_means = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    filtered_covs = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    innovations = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    innovation_covs = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    gains = tf.TensorArray(dtype=tf.float32, size=num_timesteps)

    mean_pred = initial_mean
    cov_pred = initial_cov

    identity = tf.eye(model.state_dim, dtype=tf.float32)

    for t in tf.range(num_timesteps):
        obs_t = observations[t]
        control_t = controls[t] if controls is not None else None

        mean_pred, cov_pred = _predict_step(
            model,
            mean_pred,
            cov_pred,
            control_t,
        )
        (innovation, innovation_cov, gain, mean_filt, cov_filt) = _update_step(
            model,
            mean_pred,
            cov_pred,
            obs_t,
            control_t,
            identity,
        )

        filtered_means = filtered_means.write(t, mean_filt)
        filtered_covs = filtered_covs.write(t, cov_filt)
        innovations = innovations.write(t, innovation)
        innovation_covs = innovation_covs.write(t, innovation_cov)
        gains = gains.write(t, gain)

        mean_pred = mean_filt
        cov_pred = cov_filt

    return KalmanFilterResult(
        filtered_means=filtered_means.stack(),
        filtered_covs=filtered_covs.stack(),
        innovations=innovations.stack(),
        innovation_covs=innovation_covs.stack(),
        gains=gains.stack(),
    )


def _predict_step(
    model: LinearGaussianSSM,
    mean: tf.Tensor,
    cov: tf.Tensor,
    control: Optional[tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    mean_pred = tf.linalg.matvec(model.transition_matrix, mean)
    if control is not None and model.control_matrix is not None:
        mean_pred += tf.linalg.matvec(model.control_matrix, control)
    cov_pred = (
        model.transition_matrix
        @ cov
        @ tf.transpose(model.transition_matrix)
        + model.transition_cov
    )
    return mean_pred, cov_pred


def _update_step(
    model: LinearGaussianSSM,
    mean_pred: tf.Tensor,
    cov_pred: tf.Tensor,
    observation: tf.Tensor,
    control: Optional[tf.Tensor],
    identity: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    expected_obs = tf.linalg.matvec(model.observation_matrix, mean_pred)
    if control is not None and model.observation_control_matrix is not None:
        expected_obs += tf.linalg.matvec(
            model.observation_control_matrix,
            control,
        )

    innovation = observation - expected_obs
    innovation_cov = (
        model.observation_matrix
        @ cov_pred
        @ tf.transpose(model.observation_matrix)
        + model.observation_cov
    )

    gain = cov_pred @ tf.transpose(model.observation_matrix) @ tf.linalg.inv(
        innovation_cov
    )

    mean_filt = mean_pred + tf.linalg.matvec(gain, innovation)
    joseph_factor = identity - tf.linalg.matmul(gain, model.observation_matrix)
    cov_filt = (
        tf.linalg.matmul(
            tf.linalg.matmul(joseph_factor, cov_pred),
            tf.transpose(joseph_factor),
        )
        + tf.linalg.matmul(
            tf.linalg.matmul(gain, model.observation_cov),
            tf.transpose(gain),
        )
    )

    return innovation, innovation_cov, gain, mean_filt, cov_filt
