"""Extended Kalman filter implementation using TensorFlow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel

JacobianFn = Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]


@dataclass
class ExtendedKalmanFilterResult:
    """Outputs of the extended Kalman filter."""

    filtered_means: tf.Tensor
    filtered_covs: tf.Tensor
    predicted_means: tf.Tensor
    predicted_covs: tf.Tensor
    innovations: tf.Tensor
    innovation_covs: tf.Tensor
    gains: tf.Tensor


def extended_kalman_filter(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
    initial_mean: tf.Tensor,
    initial_cov: tf.Tensor,
    controls: Optional[tf.Tensor] = None,
    transition_jacobian_fn: Optional[JacobianFn] = None,
    observation_jacobian_fn: Optional[JacobianFn] = None,
) -> ExtendedKalmanFilterResult:
    """Run the extended Kalman filter (EKF)."""

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
        if model.control_dim is None:
            raise ValueError("Controls provided but model.control_dim is None")
        expected_shape = (
            observations.shape[0],
            model.control_dim,
        )
        if controls.shape != expected_shape:
            raise ValueError(
                "controls must have shape "
                f"{expected_shape}, got {controls.shape}"
            )
    elif model.control_dim is not None:
        raise ValueError(
            "model.control_dim is set but controls were not provided"
        )

    num_timesteps = observations.shape[0]

    filtered_means = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    filtered_covs = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    predicted_means = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    predicted_covs = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    innovations = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    innovation_covs = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    gains = tf.TensorArray(dtype=tf.float32, size=num_timesteps)

    mean_prev = initial_mean
    cov_prev = initial_cov

    identity = tf.eye(model.state_dim, dtype=tf.float32)

    for t in tf.range(num_timesteps):
        control_t = controls[t] if controls is not None else None

        mean_pred, cov_pred, transition_jac = _predict_step(
            model,
            mean_prev,
            cov_prev,
            control_t,
            transition_jacobian_fn,
        )

        obs_t = observations[t]
        (
            innovation,
            innovation_cov,
            gain,
            mean_filt,
            cov_filt,
            observation_jac,
        ) = _update_step(
            model,
            mean_pred,
            cov_pred,
            obs_t,
            control_t,
            identity,
            observation_jacobian_fn,
        )

        predicted_means = predicted_means.write(t, mean_pred)
        predicted_covs = predicted_covs.write(t, cov_pred)
        filtered_means = filtered_means.write(t, mean_filt)
        filtered_covs = filtered_covs.write(t, cov_filt)
        innovations = innovations.write(t, innovation)
        innovation_covs = innovation_covs.write(t, innovation_cov)
        gains = gains.write(t, gain)

        mean_prev = mean_filt
        cov_prev = cov_filt

    return ExtendedKalmanFilterResult(
        filtered_means=filtered_means.stack(),
        filtered_covs=filtered_covs.stack(),
        predicted_means=predicted_means.stack(),
        predicted_covs=predicted_covs.stack(),
        innovations=innovations.stack(),
        innovation_covs=innovation_covs.stack(),
        gains=gains.stack(),
    )


def _predict_step(
    model: NonlinearStateSpaceModel,
    mean_prev: tf.Tensor,
    cov_prev: tf.Tensor,
    control: Optional[tf.Tensor],
    transition_jacobian_fn: Optional[JacobianFn],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    transition_jac = _compute_jacobian(
        model.transition_fn,
        mean_prev,
        control,
        transition_jacobian_fn,
    )
    mean_pred = tf.convert_to_tensor(
        model.transition_fn(mean_prev, control),
        dtype=tf.float32,
    )
    cov_pred = (
        transition_jac
        @ cov_prev
        @ tf.transpose(transition_jac)
        + model.process_noise_cov
    )
    return mean_pred, cov_pred, transition_jac


def _update_step(
    model: NonlinearStateSpaceModel,
    mean_pred: tf.Tensor,
    cov_pred: tf.Tensor,
    observation: tf.Tensor,
    control: Optional[tf.Tensor],
    identity: tf.Tensor,
    observation_jacobian_fn: Optional[JacobianFn],
) -> tuple[
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
]:
    observation_jac = _compute_jacobian(
        model.observation_fn,
        mean_pred,
        control,
        observation_jacobian_fn,
    )
    expected_obs = tf.convert_to_tensor(
        model.observation_fn(mean_pred, control),
        dtype=tf.float32,
    )

    innovation = observation - expected_obs
    innovation_cov = (
        observation_jac
        @ cov_pred
        @ tf.transpose(observation_jac)
        + model.observation_noise_cov
    )

    gain = cov_pred @ tf.transpose(observation_jac) @ tf.linalg.inv(innovation_cov)

    mean_filt = mean_pred + tf.linalg.matvec(gain, innovation)
    joseph_factor = identity - tf.linalg.matmul(gain, observation_jac)
    cov_filt = (
        tf.linalg.matmul(
            tf.linalg.matmul(joseph_factor, cov_pred),
            tf.transpose(joseph_factor),
        )
        + tf.linalg.matmul(
            tf.linalg.matmul(gain, model.observation_noise_cov),
            tf.transpose(gain),
        )
    )

    return (
        innovation,
        innovation_cov,
        gain,
        mean_filt,
        cov_filt,
        observation_jac,
    )


def _compute_jacobian(
    fn: Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor],
    state: tf.Tensor,
    control: Optional[tf.Tensor],
    jacobian_fn: Optional[JacobianFn],
) -> tf.Tensor:
    if jacobian_fn is not None:
        return tf.convert_to_tensor(jacobian_fn(state, control), dtype=tf.float32)

    state = tf.convert_to_tensor(state, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(state)
        output = fn(state, control)
    jacobian = tape.jacobian(output, state)

    return tf.convert_to_tensor(jacobian, dtype=tf.float32)
