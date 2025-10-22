"""Differentiable particle filter with entropy-regularized OT resampling."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import tensorflow as tf

from mlcoe_q2.datasets.nonlinear_ssm import NonlinearStateSpaceModel

LogLikelihoodFn = Callable[[tf.Tensor, tf.Tensor, Optional[tf.Tensor]], tf.Tensor]


@dataclass
class DifferentiablePFResult:
    """Outputs produced by the differentiable particle filter."""

    particles: tf.Tensor
    weights: tf.Tensor
    log_weights: tf.Tensor
    transport_plans: tf.Tensor
    log_likelihood: tf.Tensor
    diagnostics: dict[str, tf.Tensor]



def differentiable_particle_filter(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
    num_particles: int,
    initial_particles: tf.Tensor,
    initial_log_weights: Optional[tf.Tensor] = None,
    controls: Optional[tf.Tensor] = None,
    log_likelihood_fn: Optional[LogLikelihoodFn] = None,
    mix_with_uniform: float = 0.1,
    ot_epsilon: float = 0.1,
    ot_num_iters: int = 20,
    epsilon_schedule: Optional[Sequence[float]] = None,
    sinkhorn_tolerance: float = 1e-3,
    resampling_method: str = "ot",
) -> DifferentiablePFResult:
    """Run a differentiable particle filter with OT-based resampling."""

    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    initial_particles = tf.convert_to_tensor(initial_particles, dtype=tf.float32)

    if initial_particles.shape[0] != num_particles:
        raise ValueError(
            "initial_particles must have shape [num_particles, state_dim]"
        )

    if controls is not None:
        controls = tf.convert_to_tensor(controls, dtype=tf.float32)
        expected_shape = (
            observations.shape[0],
            model.control_dim or 0,
        )
        if controls.shape != expected_shape:
            raise ValueError(
                "controls must have shape "
                f"{expected_shape}, got {controls.shape}"
            )
    elif model.control_dim is not None:
        raise ValueError("model.control_dim is set but controls were not provided")

    if log_likelihood_fn is None:
        chol_obs_cov = tf.linalg.cholesky(model.observation_noise_cov)
        log_two_pi = tf.math.log(
            tf.constant(2.0 * math.pi, dtype=tf.float32)
        )
        dim_term = tf.cast(model.observation_dim, tf.float32) * log_two_pi

        def gaussian_log_likelihood(
            obs: tf.Tensor,
            state: tf.Tensor,
            control: Optional[tf.Tensor],
        ) -> tf.Tensor:
            expected_obs = tf.convert_to_tensor(
                model.observation_fn(state, control),
                dtype=tf.float32,
            )
            residual = obs - expected_obs
            solved = tf.linalg.cholesky_solve(
                chol_obs_cov,
                residual[:, tf.newaxis],
            )
            maha = tf.reduce_sum(
                residual * tf.squeeze(solved, axis=-1),
                axis=-1,
            )
            log_det = 2.0 * tf.reduce_sum(
                tf.math.log(tf.linalg.diag_part(chol_obs_cov))
            )
            return -0.5 * (maha + log_det + dim_term)

        log_likelihood_fn = gaussian_log_likelihood

    if initial_log_weights is None:
        log_weights = tf.zeros((num_particles,), dtype=tf.float32)
    else:
        log_weights = tf.convert_to_tensor(initial_log_weights, dtype=tf.float32)
        if log_weights.shape != (num_particles,):
            raise ValueError("initial_log_weights must have shape [num_particles]")

    num_timesteps_tensor = tf.shape(observations)[0]
    num_timesteps_static = observations.shape[0]
    num_timesteps = (
        int(num_timesteps_static) if num_timesteps_static is not None else None
    )

    if num_timesteps is not None:
        particles_size = num_timesteps + 1
        flow_steps_size = num_timesteps
    else:
        particles_size = tf.cast(num_timesteps_tensor + 1, tf.int32)
        flow_steps_size = tf.cast(num_timesteps_tensor, tf.int32)

    epsilon_schedule_tensor = None
    if epsilon_schedule is not None:
        epsilon_schedule_tensor = tf.convert_to_tensor(
            epsilon_schedule,
            dtype=tf.float32,
        )
        if epsilon_schedule_tensor.shape.rank != 1:
            raise ValueError("epsilon_schedule must be a 1-D sequence")
        if num_timesteps_static is not None and epsilon_schedule_tensor.shape[0] != num_timesteps_static:
            raise ValueError("epsilon_schedule length must match number of timesteps")

    particles_ta = tf.TensorArray(
        dtype=tf.float32,
        size=particles_size,
        clear_after_read=False,
    )
    weights_ta = tf.TensorArray(
        dtype=tf.float32,
        size=particles_size,
        clear_after_read=False,
    )
    log_weights_ta = tf.TensorArray(
        dtype=tf.float32,
        size=particles_size,
        clear_after_read=False,
    )
    transport_ta = tf.TensorArray(
        dtype=tf.float32,
        size=flow_steps_size,
    )

    particles_ta = particles_ta.write(0, initial_particles)
    normalized_weights = _normalize_log_weights(log_weights)
    weights_ta = weights_ta.write(0, normalized_weights)
    log_weights_ta = log_weights_ta.write(0, log_weights)
    log_likelihood = tf.zeros((), dtype=tf.float32)

    current_particles = initial_particles
    current_log_weights = log_weights

    diagnostics = {
        "ess": tf.TensorArray(
            dtype=tf.float32,
            size=flow_steps_size,
        ),
        "mix_weight": tf.TensorArray(
            dtype=tf.float32,
            size=flow_steps_size,
        ),
        "epsilon": tf.TensorArray(
            dtype=tf.float32,
            size=flow_steps_size,
        ),
    }

    uniform_weights = tf.fill(
        (num_particles,),
        1.0 / tf.cast(num_particles, tf.float32),
    )
    mix_factor = tf.clip_by_value(mix_with_uniform, 0.0, 1.0)

    time_indices = (
        range(num_timesteps) if num_timesteps is not None else tf.range(flow_steps_size)
    )

    for t in time_indices:
        index = t if isinstance(t, tf.Tensor) else tf.constant(t)
        control_t = (
            controls[index] if controls is not None else None
        )
        obs_t = observations[index]

        propagated_particles = _propagate(
            model,
            current_particles,
            control_t,
        )
        log_update = _update_log_weights(
            log_likelihood_fn,
            propagated_particles,
            obs_t,
            control_t,
        )
        log_weights = current_log_weights + log_update
        normalized_weights = _normalize_log_weights(log_weights)

        ess = _effective_sample_size(normalized_weights)
        diagnostics["ess"] = diagnostics["ess"].write(index, ess)
        diagnostics["mix_weight"] = diagnostics["mix_weight"].write(
            index,
            mix_factor,
        )

        mixed_weights = ((1.0 - mix_factor) * normalized_weights) + (
            mix_factor * uniform_weights
        )
        mixed_weights = _normalize_weights(mixed_weights)

        epsilon_value = (
            epsilon_schedule_tensor[index]
            if epsilon_schedule_tensor is not None
            else tf.convert_to_tensor(ot_epsilon, dtype=tf.float32)
        )
        diagnostics["epsilon"] = diagnostics["epsilon"].write(index, epsilon_value)

        if resampling_method == "ot":
            transport_plan = _entropy_regularized_transport(
                mixed_weights,
                propagated_particles,
                epsilon_value,
                ot_num_iters,
                sinkhorn_tolerance,
            )
            resampled_particles = tf.matmul(transport_plan, propagated_particles)
            resampled_weights = uniform_weights
            resampled_log_weights = tf.math.log(resampled_weights)
        elif resampling_method == "ot_low":
            low_iters = max(1, min(int(ot_num_iters), 5))
            transport_plan = _entropy_regularized_transport(
                mixed_weights,
                propagated_particles,
                epsilon_value,
                low_iters,
                sinkhorn_tolerance,
            )
            resampled_particles = tf.matmul(transport_plan, propagated_particles)
            resampled_weights = uniform_weights
            resampled_log_weights = tf.math.log(resampled_weights)
        elif resampling_method == "soft":
            # Baseline: keep particles, use mixed soft weights (no transport)
            transport_plan = tf.eye(num_particles, dtype=tf.float32)
            resampled_particles = propagated_particles
            resampled_weights = mixed_weights
            resampled_log_weights = tf.math.log(tf.maximum(resampled_weights, 1e-12))
        else:
            raise ValueError("Unknown resampling_method: " + str(resampling_method))

        transport_ta = transport_ta.write(index, transport_plan)
        particles_ta = particles_ta.write(index + 1, resampled_particles)
        weights_ta = weights_ta.write(index + 1, resampled_weights)
        log_weights_ta = log_weights_ta.write(index + 1, resampled_log_weights)

        incremental_log_likelihood = _log_sum_exp(log_weights) - tf.math.log(
            tf.cast(num_particles, tf.float32)
        )
        log_likelihood = log_likelihood + incremental_log_likelihood

        current_particles = resampled_particles
        current_log_weights = resampled_log_weights

    transport_plans = transport_ta.stack()
    diagnostics = {key: value.stack() for key, value in diagnostics.items()}

    return DifferentiablePFResult(
        particles=particles_ta.stack(),
        weights=weights_ta.stack(),
        log_weights=log_weights_ta.stack(),
        transport_plans=transport_plans,
        log_likelihood=log_likelihood,
        diagnostics=diagnostics,
    )





def _propagate(
    model: NonlinearStateSpaceModel,
    particles: tf.Tensor,
    control: Optional[tf.Tensor],
) -> tf.Tensor:
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


def _update_log_weights(
    log_likelihood_fn: LogLikelihoodFn,
    particles: tf.Tensor,
    observation: tf.Tensor,
    control: Optional[tf.Tensor],
) -> tf.Tensor:
    def body(particle: tf.Tensor) -> tf.Tensor:
        return log_likelihood_fn(observation, particle, control)

    return tf.map_fn(body, particles, fn_output_signature=tf.float32)


def _normalize_log_weights(log_weights: tf.Tensor) -> tf.Tensor:
    max_val = tf.reduce_max(log_weights)
    stabilized = tf.exp(log_weights - max_val)
    normalized = stabilized / tf.reduce_sum(stabilized)
    return normalized


def _normalize_weights(weights: tf.Tensor) -> tf.Tensor:
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    total = tf.reduce_sum(weights)
    num = tf.cast(tf.shape(weights)[0], tf.float32)
    total = tf.where(total > 0.0, total, num)
    return tf.where(
        total > 0.0,
        weights / total,
        tf.fill(tf.shape(weights), 1.0 / num),
    )


def _effective_sample_size(weights: tf.Tensor) -> tf.Tensor:
    return 1.0 / tf.reduce_sum(tf.square(weights) + 1e-12)


def _log_sum_exp(log_weights: tf.Tensor) -> tf.Tensor:
    max_val = tf.reduce_max(log_weights)
    return max_val + tf.math.log(tf.reduce_sum(tf.exp(log_weights - max_val)))


def _entropy_regularized_transport(
    source_weights: tf.Tensor,
    particles: tf.Tensor,
    epsilon: tf.Tensor,
    num_iters: int,
    tolerance: float,
) -> tf.Tensor:
    source_weights = _normalize_weights(source_weights)
    num_particles = tf.shape(particles)[0]
    target_weights = tf.fill(
        (num_particles,),
        1.0 / tf.cast(num_particles, tf.float32),
    )

    epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
    cost_matrix = _pairwise_squared_distances(particles)
    kernel = tf.exp(-cost_matrix / tf.maximum(epsilon, 1e-6))

    eps = tf.constant(1e-8, dtype=tf.float32)
    u = tf.ones_like(source_weights) / tf.cast(num_particles, tf.float32)
    v = tf.ones_like(target_weights) / tf.cast(num_particles, tf.float32)

    for _ in range(max(num_iters, 1)):
        Kv = tf.matmul(kernel, v[:, tf.newaxis])
        u = source_weights / tf.maximum(tf.squeeze(Kv, axis=-1), eps)
        Ku = tf.matmul(tf.transpose(kernel), u[:, tf.newaxis])
        v = target_weights / tf.maximum(tf.squeeze(Ku, axis=-1), eps)

        diag_u = tf.linalg.diag(u)
        diag_v = tf.linalg.diag(v)
        transport = tf.matmul(diag_u, tf.matmul(kernel, diag_v))
        if tolerance > 0.0 and tf.executing_eagerly():
            row_error = tf.reduce_max(
                tf.abs(tf.reduce_sum(transport, axis=1) - source_weights)
            )
            col_error = tf.reduce_max(
                tf.abs(tf.reduce_sum(transport, axis=0) - target_weights)
            )
            if float(row_error.numpy()) < tolerance and float(col_error.numpy()) < tolerance:
                break
    else:
        transport = tf.matmul(tf.linalg.diag(u), tf.matmul(kernel, tf.linalg.diag(v)))

    row_sums = tf.reduce_sum(transport, axis=1, keepdims=True)
    transport = transport / tf.maximum(row_sums, eps)
    return transport


def _pairwise_squared_distances(particles: tf.Tensor) -> tf.Tensor:
    diffs = particles[:, tf.newaxis, :] - particles[tf.newaxis, :, :]
    return tf.reduce_sum(tf.square(diffs), axis=-1)
