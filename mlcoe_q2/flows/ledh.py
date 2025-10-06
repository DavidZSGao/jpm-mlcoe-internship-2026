
"""Local Exact Daum-Huang (LEDH) particle flow implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import tensorflow as tf

from mlcoe_q2.datasets.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.flows.base import ParticleFlowResult
from mlcoe_q2.flows.edh import _normalize_weights, _weighted_covariance


@dataclass
class LocalExactDaumHuangFlow:
    """LEDH flow using per-particle linearizations of the observation model."""

    step_size: float = 1.0
    num_steps: int = 3
    jitter: float = 1e-5

    def __call__(
        self,
        model: NonlinearStateSpaceModel,
        particles: tf.Tensor,
        weights: tf.Tensor,
        observation: tf.Tensor,
        control: Optional[tf.Tensor] = None,
    ) -> ParticleFlowResult:
        particles = tf.convert_to_tensor(particles, dtype=tf.float32)
        weights = _normalize_weights(weights)
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        if control is not None:
            control = tf.convert_to_tensor(control, dtype=tf.float32)

        covariance = _weighted_covariance(particles, weights, self.jitter)
        obs_noise = tf.convert_to_tensor(
            model.observation_noise_cov, dtype=tf.float32
        )
        identity_obs = tf.eye(model.observation_dim, dtype=tf.float32)
        identity_state = tf.eye(model.state_dim, dtype=tf.float32)

        steps = tf.cast(max(self.num_steps, 1), tf.float32)
        step_scale = tf.cast(self.step_size, tf.float32) / steps

        diagnostic_delta_norms = []
        diagnostic_cond_numbers = []
        diagnostic_log_det = []

        updated_particles = particles
        log_jacobians = tf.zeros(tf.shape(weights), dtype=tf.float32)

        for _ in range(max(self.num_steps, 1)):
            (
                deltas,
                delta_norms,
                cond_numbers,
                log_det,
            ) = _local_flow_updates(
                model,
                updated_particles,
                observation,
                control,
                covariance,
                obs_noise,
                identity_obs,
                identity_state,
                self.jitter,
                step_scale,
            )

            diagnostic_delta_norms.append(tf.reduce_mean(delta_norms))
            diagnostic_cond_numbers.append(tf.reduce_mean(cond_numbers))
            diagnostic_log_det.append(tf.reduce_mean(log_det))

            updated_particles = updated_particles + step_scale * deltas
            log_jacobians = log_jacobians + log_det

        diagnostics = {
            "delta_norm_mean": tf.stack(diagnostic_delta_norms),
            "innovation_cond_mean": tf.stack(diagnostic_cond_numbers),
            "log_det_mean": tf.stack(diagnostic_log_det),
        }

        return ParticleFlowResult(
            propagated_particles=updated_particles,
            propagated_weights=weights,
            log_jacobians=log_jacobians,
            diagnostics=diagnostics,
        )


def _local_flow_updates(
    model: NonlinearStateSpaceModel,
    particles: tf.Tensor,
    observation: tf.Tensor,
    control: Optional[tf.Tensor],
    covariance: tf.Tensor,
    observation_noise: tf.Tensor,
    identity_obs: tf.Tensor,
    identity_state: tf.Tensor,
    jitter: float,
    step_scale: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    obs_dim = tf.shape(observation)[-1]
    jitter_eye = jitter * identity_obs
    state_dim = model.state_dim

    def body(particle: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        particle = tf.convert_to_tensor(particle, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(particle)
            obs_pred = tf.convert_to_tensor(
                model.observation_fn(particle, control),
                dtype=tf.float32,
            )
            residual = observation - obs_pred
            jacobian = tape.jacobian(obs_pred, particle)

            cov_jac_t = tf.matmul(covariance, jacobian, transpose_b=True)
            innovation_cov = (
                tf.matmul(jacobian, cov_jac_t) + observation_noise + jitter_eye
            )
            chol = tf.linalg.cholesky(innovation_cov)
            gain = tf.linalg.cholesky_solve(chol, tf.transpose(cov_jac_t))
            gain = tf.transpose(gain)

            delta = tf.matmul(gain, residual[:, tf.newaxis])
            delta = tf.reshape(delta, (-1,))

        delta_jacobian = tape.jacobian(delta, particle)
        del tape

        delta_norm = tf.linalg.norm(delta)
        singular_values = tf.linalg.svd(
            innovation_cov,
            compute_uv=False,
        )
        smallest = tf.reduce_min(singular_values)
        largest = tf.reduce_max(singular_values)
        cond_number = largest / tf.maximum(smallest, 1e-6)

        jacobian_matrix = identity_state + step_scale * tf.convert_to_tensor(
            delta_jacobian, dtype=tf.float32
        )
        jacobian_matrix = jacobian_matrix + jitter * identity_state
        sign, log_abs_det = tf.linalg.slogdet(jacobian_matrix)
        log_det = log_abs_det

        return (
            delta,
            delta_norm,
            cond_number,
            log_det,
        )

    deltas, delta_norms, cond_numbers, log_det = tf.map_fn(
        lambda x: body(x),
        particles,
        fn_output_signature=(
            tf.TensorSpec((state_dim,), tf.float32),
            tf.TensorSpec((), tf.float32),
            tf.TensorSpec((), tf.float32),
            tf.TensorSpec((), tf.float32),
        ),
    )

    return deltas, delta_norms, cond_numbers, log_det
