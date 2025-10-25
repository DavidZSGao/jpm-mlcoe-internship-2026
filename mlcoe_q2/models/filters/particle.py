"""Particle filter implementation for nonlinear state-space models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel

LogLikelihoodFn = Callable[[tf.Tensor, tf.Tensor, Optional[tf.Tensor]], tf.Tensor]


@dataclass
class ParticleFilterResult:
    """Outputs produced by the particle filter."""

    particles: tf.Tensor
    weights: tf.Tensor
    log_weights: tf.Tensor
    ancestor_indices: tf.Tensor
    effective_sample_sizes: tf.Tensor
    log_likelihood: tf.Tensor


def particle_filter(
    model: NonlinearStateSpaceModel,
    observations: tf.Tensor,
    num_particles: int,
    initial_particles: tf.Tensor,
    initial_log_weights: Optional[tf.Tensor] = None,
    controls: Optional[tf.Tensor] = None,
    log_likelihood_fn: Optional[LogLikelihoodFn] = None,
    resample_threshold: float = 0.5,
    resample_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
) -> ParticleFilterResult:
    """Run a Sequential Importance Resampling (SIR) particle filter.

    Args:
        model: Nonlinear state-space model defining transition/observation maps.
        observations: Sequence of observed measurements, shape `[T, obs_dim]`.
        num_particles: Number of particles to propagate.
        initial_particles: Initial particle states, shape `[num_particles, state_dim]`.
        initial_log_weights: Optional log-weights, shape `[num_particles]`. Defaults to
            uniform weights.
        controls: Optional control inputs, shape `[T, control_dim]`.
        log_likelihood_fn: Callable returning log-likelihood of an observation given
            state and (optional) control. If omitted, assumes additive Gaussian noise
            from `model.observation_noise_cov`.
        resample_threshold: ESS threshold in (0, 1]. When ESS / num_particles drops
            below this value, particles are resampled.
        resample_fn: Optional custom resampling function. Defaults to systematic
            resampling.

    Returns:
        `ParticleFilterResult` containing particle trajectories, weights, ancestor
        indices, ESS sequence, and marginal log-likelihood estimate.
    """

    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    initial_particles = tf.convert_to_tensor(initial_particles, dtype=tf.float32)
    if initial_particles.shape[0] != num_particles:
        raise ValueError(
            "initial_particles must have shape [num_particles, state_dim]"
        )

    if observations.shape[-1] != model.observation_dim:
        raise ValueError("Observation dimension mismatch")
    if initial_particles.shape[-1] != model.state_dim:
        raise ValueError("Particle state dimension mismatch")
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")

    if initial_log_weights is None:
        log_weights = tf.zeros((num_particles,), dtype=tf.float32)
    else:
        log_weights = tf.convert_to_tensor(initial_log_weights, dtype=tf.float32)
        if log_weights.shape != (num_particles,):
            raise ValueError("initial_log_weights must have shape [num_particles]")

    if controls is not None:
        controls = tf.convert_to_tensor(controls, dtype=tf.float32)
        if model.control_dim is None:
            raise ValueError("Controls provided but model.control_dim is None")
        expected_shape = (observations.shape[0], model.control_dim)
        if controls.shape != expected_shape:
            raise ValueError(
                "controls must have shape "
                f"{expected_shape}, got {controls.shape}"
            )
    elif model.control_dim is not None:
        raise ValueError("model.control_dim is set but controls were not provided")

    if resample_fn is None:
        resample_fn = _systematic_resample

    if log_likelihood_fn is None:
        chol_obs_cov = tf.linalg.cholesky(model.observation_noise_cov)

        log_two_pi = tf.math.log(tf.constant(2.0 * math.pi, dtype=tf.float32))
        dim_term = tf.cast(model.observation_dim, tf.float32) * log_two_pi

        def gaussian_log_likelihood(
            obs: tf.Tensor, state: tf.Tensor, control: Optional[tf.Tensor]
        ) -> tf.Tensor:
            expected_obs = tf.convert_to_tensor(
                model.observation_fn(state, control), dtype=tf.float32
            )
            residual = obs - expected_obs
            solved = tf.linalg.triangular_solve(
                chol_obs_cov,
                residual[..., tf.newaxis],
                lower=True,
            )
            maha = tf.reduce_sum(tf.squeeze(solved, axis=-1) ** 2, axis=-1)
            log_det = 2.0 * tf.reduce_sum(
                tf.math.log(tf.linalg.diag_part(chol_obs_cov))
            )
            return -0.5 * (maha + log_det + dim_term)

        log_likelihood_fn = gaussian_log_likelihood

    num_timesteps = observations.shape[0]
    state_dim = model.state_dim

    particles_ta = tf.TensorArray(dtype=tf.float32, size=num_timesteps + 1)
    weights_ta = tf.TensorArray(dtype=tf.float32, size=num_timesteps + 1)
    log_weights_ta = tf.TensorArray(dtype=tf.float32, size=num_timesteps + 1)
    ancestors_ta = tf.TensorArray(dtype=tf.int32, size=num_timesteps)
    ess_ta = tf.TensorArray(dtype=tf.float32, size=num_timesteps)

    particles_ta = particles_ta.write(0, initial_particles)
    normalized_weights = _normalize_log_weights(log_weights)
    weights_ta = weights_ta.write(0, normalized_weights)
    log_weights_ta = log_weights_ta.write(0, log_weights)
    log_likelihood = tf.zeros((), dtype=tf.float32)

    current_particles = initial_particles
    current_log_weights = log_weights

    threshold = tf.cast(resample_threshold, tf.float32)
    particle_count_f = tf.cast(num_particles, tf.float32)
    uniform_weights = tf.fill((num_particles,), 1.0 / particle_count_f)

    for t in tf.range(num_timesteps):
        control_t = controls[t] if controls is not None else None
        obs_t = observations[t]

        propagated_particles = _propagate(
            model,
            current_particles,
            control_t,
        )

        log_weights = _update_log_weights(
            log_likelihood_fn,
            propagated_particles,
            obs_t,
            control_t,
        )
        log_weights = log_weights + current_log_weights

        normalized_weights = _normalize_log_weights(log_weights)
        effective_sample_size = _effective_sample_size(normalized_weights)

        should_resample = tf.math.logical_and(
            threshold > 0.0,
            effective_sample_size
            <= threshold * particle_count_f,
        )

        resampled_indices = tf.cond(
            should_resample,
            lambda: resample_fn(normalized_weights),
            lambda: tf.range(num_particles, dtype=tf.int32),
        )

        post_particles = tf.cond(
            should_resample,
            lambda: tf.gather(propagated_particles, resampled_indices),
            lambda: propagated_particles,
        )
        post_log_weights = tf.cond(
            should_resample,
            lambda: tf.zeros_like(log_weights),
            lambda: log_weights,
        )
        post_normalized_weights = tf.cond(
            should_resample,
            lambda: uniform_weights,
            lambda: normalized_weights,
        )

        ess_ta = ess_ta.write(t, effective_sample_size)
        ancestors_ta = ancestors_ta.write(t, resampled_indices)
        particles_ta = particles_ta.write(t + 1, post_particles)
        weights_ta = weights_ta.write(t + 1, post_normalized_weights)
        log_weights_ta = log_weights_ta.write(t + 1, post_log_weights)

        incremental_log_likelihood = _log_sum_exp(log_weights) - tf.math.log(
            particle_count_f
        )
        log_likelihood = log_likelihood + incremental_log_likelihood

        current_particles = post_particles
        current_log_weights = post_log_weights

    return ParticleFilterResult(
        particles=particles_ta.stack(),
        weights=weights_ta.stack(),
        log_weights=log_weights_ta.stack(),
        ancestor_indices=ancestors_ta.stack(),
        effective_sample_sizes=ess_ta.stack(),
        log_likelihood=log_likelihood,
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
            model.transition_fn(particle, control), dtype=tf.float32
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
    log_normalizer = _log_sum_exp(log_weights)
    return tf.nn.softmax(log_weights - log_normalizer)


def _effective_sample_size(normalized_weights: tf.Tensor) -> tf.Tensor:
    return 1.0 / tf.reduce_sum(normalized_weights ** 2)


def _log_sum_exp(values: tf.Tensor) -> tf.Tensor:
    max_val = tf.reduce_max(values)
    stabilized = tf.exp(values - max_val)
    return tf.math.log(tf.reduce_sum(stabilized)) + max_val


def _systematic_resample(weights: tf.Tensor) -> tf.Tensor:
    num_particles = tf.shape(weights)[0]
    step = 1.0 / tf.cast(num_particles, tf.float32)
    base = tf.random.uniform((), dtype=tf.float32, maxval=step)
    positions = base + step * tf.cast(tf.range(num_particles), tf.float32)
    cumsum = tf.cumsum(weights)
    indices = tf.searchsorted(cumsum, positions, side="right")
    indices = tf.cast(tf.clip_by_value(indices, 0, num_particles - 1), tf.int32)
    return indices
