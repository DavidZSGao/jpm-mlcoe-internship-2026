"""Unscented Kalman filter (UKF) implementation in TensorFlow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel


@dataclass
class UnscentedKalmanFilterResult:
    """Outputs produced by the unscented Kalman filter."""

    filtered_means: tf.Tensor
    filtered_covs: tf.Tensor
    predicted_means: tf.Tensor
    predicted_covs: tf.Tensor
    innovations: tf.Tensor
    innovation_covs: tf.Tensor
    gains: tf.Tensor


def unscented_kalman_filter(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
    initial_mean: tf.Tensor,
    initial_cov: tf.Tensor,
    controls: Optional[tf.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> UnscentedKalmanFilterResult:
    """Run the unscented Kalman filter (UKF)."""

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
    state_dim = model.state_dim
    obs_dim = model.observation_dim
    aug_dim = state_dim + state_dim + obs_dim

    weights_mean, weights_cov, lambda_val = _unscented_weights(
        aug_dim,
        alpha,
        beta,
        kappa,
    )

    filtered_means = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    filtered_covs = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    predicted_means = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    predicted_covs = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    innovations = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    innovation_covs = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
    gains = tf.TensorArray(dtype=tf.float32, size=num_timesteps)

    mean_prev = initial_mean
    cov_prev = initial_cov

    for t in tf.range(num_timesteps):
        control_t = controls[t] if controls is not None else None

        augmented_mean = tf.concat(
            [
                mean_prev,
                tf.zeros(state_dim, dtype=tf.float32),
                tf.zeros(obs_dim, dtype=tf.float32),
            ],
            axis=0,
        )
        zeros_x = tf.zeros((state_dim, state_dim), dtype=tf.float32)
        zeros_y = tf.zeros((state_dim, obs_dim), dtype=tf.float32)
        zeros_z = tf.zeros((obs_dim, state_dim), dtype=tf.float32)

        top_row = tf.concat([cov_prev, zeros_x, zeros_y], axis=1)
        middle_row = tf.concat([zeros_x, model.process_noise_cov, zeros_y], axis=1)
        bottom_row = tf.concat([zeros_z, zeros_z, model.observation_noise_cov], axis=1)

        augmented_cov = tf.concat([top_row, middle_row, bottom_row], axis=0)
        sigma_points = _compute_sigma_points(
            augmented_mean,
            augmented_cov,
            lambda_val,
        )
        state_sigma, process_sigma, measurement_sigma = tf.split(
            sigma_points,
            [state_dim, state_dim, obs_dim],
            axis=-1,
        )

        num_sigma = tf.shape(sigma_points)[0]
        state_array = tf.TensorArray(dtype=tf.float32, size=num_sigma)
        for i in tf.range(num_sigma):
            state_i = state_sigma[i]
            noise_i = process_sigma[i]
            transitioned = tf.convert_to_tensor(
                model.transition_fn(state_i, control_t),
                dtype=tf.float32,
            )
            state_array = state_array.write(i, transitioned + noise_i)
        propagated_sigma = state_array.stack()

        obs_array = tf.TensorArray(dtype=tf.float32, size=num_sigma)
        for i in tf.range(num_sigma):
            obs_state = propagated_sigma[i]
            noise_i = measurement_sigma[i]
            observed = tf.convert_to_tensor(
                model.observation_fn(obs_state, control_t),
                dtype=tf.float32,
            )
            obs_array = obs_array.write(i, observed + noise_i)
        obs_sigma = obs_array.stack()

        mean_pred = _weighted_mean(propagated_sigma, weights_mean)
        cov_pred = _weighted_cov(
            propagated_sigma,
            mean_pred,
            weights_cov,
        )

        obs_mean = _weighted_mean(obs_sigma, weights_mean)
        obs_cov = _weighted_cov(
            obs_sigma,
            obs_mean,
            weights_cov,
        )

        cross_cov = _weighted_cross_cov(
            propagated_sigma,
            mean_pred,
            obs_sigma,
            obs_mean,
            weights_cov,
        )

        gain = tf.linalg.solve(obs_cov, tf.transpose(cross_cov))
        gain = tf.transpose(gain)

        innovation = observations[t] - obs_mean
        mean_filt = mean_pred + tf.linalg.matvec(gain, innovation)
        cov_filt = cov_pred - tf.linalg.matmul(
            tf.linalg.matmul(gain, obs_cov),
            tf.transpose(gain),
        )

        predicted_means = predicted_means.write(t, mean_pred)
        predicted_covs = predicted_covs.write(t, cov_pred)
        filtered_means = filtered_means.write(t, mean_filt)
        filtered_covs = filtered_covs.write(t, cov_filt)
        innovations = innovations.write(t, innovation)
        innovation_covs = innovation_covs.write(t, obs_cov)
        gains = gains.write(t, gain)

        mean_prev = mean_filt
        cov_prev = cov_filt

    return UnscentedKalmanFilterResult(
        filtered_means=filtered_means.stack(),
        filtered_covs=filtered_covs.stack(),
        predicted_means=predicted_means.stack(),
        predicted_covs=predicted_covs.stack(),
        innovations=innovations.stack(),
        innovation_covs=innovation_covs.stack(),
        gains=gains.stack(),
    )


def _unscented_weights(
    state_dim: int,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    state_dim_f = tf.cast(state_dim, tf.float32)
    alpha_sq = tf.cast(alpha, tf.float32) ** 2
    lambda_val = alpha_sq * (state_dim_f + tf.cast(kappa, tf.float32)) - state_dim_f
    denom = state_dim_f + lambda_val

    w0_mean = lambda_val / denom
    w0_cov = w0_mean + (1.0 - alpha_sq + tf.cast(beta, tf.float32))
    wi = 0.5 / denom

    tail = tf.fill((2 * state_dim,), wi)
    weights_mean = tf.concat([[w0_mean], tail], axis=0)
    weights_cov = tf.concat([[w0_cov], tail], axis=0)

    return weights_mean, weights_cov, lambda_val


def _compute_sigma_points(
    mean: tf.Tensor,
    cov: tf.Tensor,
    lambda_val: tf.Tensor,
) -> tf.Tensor:
    state_dim = mean.shape[-1]
    scaling = tf.sqrt(tf.cast(state_dim, tf.float32) + lambda_val)
    chol = tf.linalg.cholesky(cov)
    scaled = chol * scaling

    sigma_list = [mean]
    for i in range(state_dim):
        offset = scaled[:, i]
        sigma_list.append(mean + offset)
        sigma_list.append(mean - offset)
    return tf.stack(sigma_list, axis=0)


def _propagate_sigma_points(
    sigma_points: tf.Tensor,
    fn,
    control: Optional[tf.Tensor],
) -> tf.Tensor:
    def body(point: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(fn(point, control), dtype=tf.float32)

    return tf.map_fn(body, sigma_points, fn_output_signature=tf.float32)


def _weighted_mean(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.tensordot(weights, points, axes=1)


def _weighted_cov(
    points: tf.Tensor,
    mean: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    diff = points - mean[tf.newaxis, :]
    return tf.einsum("i,ij,ik->jk", weights, diff, diff)


def _weighted_cross_cov(
    state_points: tf.Tensor,
    state_mean: tf.Tensor,
    obs_points: tf.Tensor,
    obs_mean: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    state_diff = state_points - state_mean[tf.newaxis, :]
    obs_diff = obs_points - obs_mean[tf.newaxis, :]
    return tf.einsum("i,ij,ik->jk", weights, state_diff, obs_diff)
