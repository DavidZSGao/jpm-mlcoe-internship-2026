
"""Stochastic particle flow following Dai (2022) style updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.models.flows.base import ParticleFlowResult
from mlcoe_q2.models.flows.edh import (
    _log_likelihood_derivatives,
    _normalize_weights,
    _weighted_covariance,
)


@dataclass
class StochasticParticleFlow:
    """Approximate stochastic particle flow integrating diffusion."""

    step_size: float = 1.0
    num_steps: int = 6
    diffusion: float = 0.1
    jitter: float = 1e-4

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

        steps = tf.cast(max(self.num_steps, 1), tf.float32)
        step_scale = tf.cast(self.step_size, tf.float32) / steps
        diffusion_coeff = tf.cast(self.diffusion, tf.float32)
        identity = tf.eye(model.state_dim, dtype=tf.float32)

        updated_particles = particles

        grad_norms = []
        diffusion_norms = []
        residual_norms = []
        log_det_terms = []

        log_jacobians = tf.zeros(tf.shape(weights), dtype=tf.float32)

        for _ in range(max(self.num_steps, 1)):
            covariance = _weighted_covariance(
                updated_particles,
                weights,
                self.jitter,
            )
            chol = tf.linalg.cholesky(covariance)

            grads, residual, hessians = _log_likelihood_derivatives(
                model,
                updated_particles,
                observation,
                control,
            )

            grad_norms.append(tf.reduce_mean(tf.linalg.norm(grads, axis=1)))
            residual_norms.append(tf.reduce_mean(residual))

            drift = tf.matmul(grads, covariance)

            noise = tf.random.normal(
                tf.shape(updated_particles),
                dtype=tf.float32,
            )
            diffusion_scale = tf.sqrt(2.0 * diffusion_coeff * step_scale)
            diffusion_step = diffusion_scale * tf.matmul(noise, chol)
            diffusion_norms.append(
                tf.reduce_mean(
                    tf.linalg.norm(diffusion_step, axis=1)
                )
            )

            updated_particles = (
                updated_particles + step_scale * drift + diffusion_step
            )

            cov_hess = tf.einsum("ij,njk->nik", covariance, hessians)
            jacobian_mats = identity + step_scale * cov_hess
            jacobian_mats = jacobian_mats + self.jitter * identity
            sign, log_abs_det = tf.linalg.slogdet(jacobian_mats)
            log_det = log_abs_det
            log_jacobians = log_jacobians + log_det
            log_det_terms.append(tf.reduce_mean(log_det))

        diagnostics = {
            "grad_norm_mean": tf.stack(grad_norms),
            "diffusion_norm_mean": tf.stack(diffusion_norms),
            "residual_norm_mean": tf.stack(residual_norms),
            "log_det_mean": tf.stack(log_det_terms),
        }

        return ParticleFlowResult(
            propagated_particles=updated_particles,
            propagated_weights=weights,
            log_jacobians=log_jacobians,
            diagnostics=diagnostics,
        )
