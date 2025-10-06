
"""Approximate Exact Daum-Huang (EDH) particle flow implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from mlcoe_q2.datasets.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.flows.base import ParticleFlowResult


@dataclass
class ExactDaumHuangFlow:
    """Approximate EDH flow using covariance-preconditioned score updates."""

    step_size: float = 1.0
    num_steps: int = 5
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

        covariance = _weighted_covariance(particles, weights, self.jitter)
        steps = tf.cast(max(self.num_steps, 1), tf.float32)
        step_scale = tf.cast(self.step_size, tf.float32) / steps
        identity = tf.eye(model.state_dim, dtype=tf.float32)

        diagnostic_grad_norms = []
        diagnostic_residual_norms = []
        diagnostic_log_det = []

        updated_particles = particles
        log_jacobians = tf.zeros(tf.shape(weights), dtype=tf.float32)

        for _ in range(max(self.num_steps, 1)):
            grads, residual_norms, hessians = _log_likelihood_derivatives(
                model,
                updated_particles,
                observation,
                control,
            )
            diagnostic_grad_norms.append(
                tf.reduce_mean(tf.norm(grads, axis=1))
            )
            diagnostic_residual_norms.append(
                tf.reduce_mean(residual_norms)
            )

            updates = tf.matmul(grads, covariance)
            updated_particles = updated_particles + step_scale * updates

            cov_hess = tf.einsum("ij,njk->nik", covariance, hessians)
            jacobian_mats = identity + step_scale * cov_hess
            jacobian_mats = jacobian_mats + self.jitter * identity
            sign, log_abs_det = tf.linalg.slogdet(jacobian_mats)
            log_det = log_abs_det
            log_jacobians = log_jacobians + log_det
            diagnostic_log_det.append(tf.reduce_mean(log_det))

        diagnostics = {
            "grad_norm_mean": tf.stack(diagnostic_grad_norms),
            "residual_norm_mean": tf.stack(diagnostic_residual_norms),
            "log_det_mean": tf.stack(diagnostic_log_det),
        }

        return ParticleFlowResult(
            propagated_particles=updated_particles,
            propagated_weights=weights,
            log_jacobians=log_jacobians,
            diagnostics=diagnostics,
        )


def _normalize_weights(weights: tf.Tensor) -> tf.Tensor:
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    if tf.rank(weights) == 2:
        weights = tf.squeeze(weights, axis=-1)
    positive = tf.maximum(weights, 0.0)
    total = tf.reduce_sum(positive)
    num = tf.cast(tf.shape(positive)[0], tf.float32)
    fallback = tf.fill(tf.shape(positive), 1.0 / num)
    normalized = tf.where(total > 0.0, positive / total, fallback)
    return normalized


def _weighted_covariance(
    particles: tf.Tensor,
    weights: tf.Tensor,
    jitter: float,
) -> tf.Tensor:
    weights = tf.reshape(weights, (-1,))
    weights = _normalize_weights(weights)
    mean = tf.reduce_sum(particles * weights[:, tf.newaxis], axis=0)
    centered = particles - mean
    cov = tf.einsum("ni,nj,n->ij", centered, centered, weights)
    cov += jitter * tf.eye(tf.shape(cov)[0], dtype=tf.float32)
    return cov


def _log_likelihood_derivatives(
    model: NonlinearStateSpaceModel,
    particles: tf.Tensor,
    observation: tf.Tensor,
    control: Optional[tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    chol = tf.linalg.cholesky(model.observation_noise_cov)
    state_dim = model.state_dim
    obs_dim = model.observation_dim

    def body(particle: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        particle = tf.convert_to_tensor(particle, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape_outer:
            tape_outer.watch(particle)
            with tf.GradientTape() as tape_inner:
                tape_inner.watch(particle)
                obs_pred = tf.convert_to_tensor(
                    model.observation_fn(particle, control),
                    dtype=tf.float32,
                )
                residual = observation - obs_pred
                solve = tf.linalg.cholesky_solve(
                    chol,
                    residual[:, tf.newaxis],
                )
                quadratic = tf.squeeze(
                    tf.matmul(residual[tf.newaxis, :], solve),
                    axis=0,
                )
                log_prob = -0.5 * quadratic
            grad = tape_inner.gradient(log_prob, particle)
        hessian = tape_outer.jacobian(grad, particle)
        residual_norm = tf.linalg.norm(residual)
        del tape_outer
        return (
            tf.reshape(grad, (state_dim,)),
            residual_norm,
            tf.reshape(hessian, (state_dim, state_dim)),
        )

    grads, residual_norms, hessians = tf.map_fn(
        lambda x: body(x),
        particles,
        fn_output_signature=(
            tf.TensorSpec((state_dim,), tf.float32),
            tf.TensorSpec((), tf.float32),
            tf.TensorSpec((state_dim, state_dim), tf.float32),
        ),
    )
    return grads, residual_norms, hessians
