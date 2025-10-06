
"""Kernel-embedded particle flow implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import tensorflow as tf

from mlcoe_q2.datasets.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.flows.base import ParticleFlowResult
from mlcoe_q2.flows.edh import _normalize_weights

KernelType = Literal["scalar", "diagonal", "matrix"]


@dataclass
class KernelEmbeddedFlow:
    """Kernel-embedded deterministic particle flow in an RKHS."""

    kernel_type: KernelType = "scalar"
    bandwidth: float = 1.0
    jitter: float = 1e-4
    step_size: float = 1.0
    num_steps: int = 3

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
        identity_state = tf.eye(model.state_dim, dtype=tf.float32)

        diagnostics = {
            "kernel_trace": [],
            "kernel_condition": [],
            "score_norm_mean": [],
            "log_det_mean": [],
        }

        updated_particles = particles
        log_jacobians = tf.zeros(tf.shape(weights), dtype=tf.float32)

        for _ in range(max(self.num_steps, 1)):
            kernel_matrix, kernel_chol, metric = _compute_kernel(
                updated_particles,
                self.bandwidth,
                self.kernel_type,
                self.jitter,
            )
            scores, score_jacobians = _kernel_scores(
                model,
                updated_particles,
                observation,
                control,
                kernel_matrix,
                kernel_chol,
                metric,
            )

            diagnostics["kernel_trace"].append(
                tf.linalg.trace(kernel_matrix)
            )
            diagnostics["kernel_condition"].append(
                _condition_number(kernel_matrix)
            )
            diagnostics["score_norm_mean"].append(
                tf.reduce_mean(tf.norm(scores, axis=1))
            )

            updated_particles = updated_particles + step_scale * scores

            jacobian_mats = identity_state + step_scale * score_jacobians
            jacobian_mats = jacobian_mats + self.jitter * identity_state
            sign, log_abs_det = tf.linalg.slogdet(jacobian_mats)
            log_det = log_abs_det
            log_jacobians = log_jacobians + log_det
            diagnostics["log_det_mean"].append(tf.reduce_mean(log_det))

        diagnostics = {
            key: tf.stack(values) for key, values in diagnostics.items()
        }

        return ParticleFlowResult(
            propagated_particles=updated_particles,
            propagated_weights=weights,
            log_jacobians=log_jacobians,
            diagnostics=diagnostics,
        )


def _compute_kernel(
    particles: tf.Tensor,
    bandwidth: float,
    kernel_type: KernelType,
    jitter: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    state_dim = tf.shape(particles)[-1]
    diffs = particles[:, tf.newaxis, :] - particles[tf.newaxis, :, :]

    if kernel_type == "scalar":
        scaled = diffs / (bandwidth + 1e-8)
        squared_dists = tf.reduce_sum(scaled**2, axis=-1)
        kernel = tf.exp(-0.5 * squared_dists)
        metric = (1.0 / (bandwidth**2 + 1e-8)) * tf.eye(
            state_dim,
            dtype=tf.float32,
        )
    elif kernel_type == "diagonal":
        variance = tf.math.reduce_variance(particles, axis=0) + 1e-6
        scales = bandwidth / variance
        scaled = diffs * tf.sqrt(scales)[tf.newaxis, tf.newaxis, :]
        squared_dists = tf.reduce_sum(scaled**2, axis=-1)
        kernel = tf.exp(-0.5 * squared_dists)
        metric = tf.linalg.diag(scales)
    elif kernel_type == "matrix":
        sample_cov = _sample_covariance(particles)
        sample_cov += jitter * tf.eye(state_dim, dtype=tf.float32)
        metric = tf.linalg.inv(sample_cov)
        metric = metric / (bandwidth + 1e-8)
        mahal = tf.einsum("...i,ij,...j->...", diffs, metric, diffs)
        kernel = tf.exp(-0.5 * mahal)
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    jitter_matrix = jitter * tf.eye(tf.shape(kernel)[0], dtype=tf.float32)
    kernel = kernel + jitter_matrix
    chol = tf.linalg.cholesky(kernel)
    return kernel, chol, tf.convert_to_tensor(metric, dtype=tf.float32)


def _kernel_scores(
    model: NonlinearStateSpaceModel,
    particles: tf.Tensor,
    observation: tf.Tensor,
    control: Optional[tf.Tensor],
    kernel_matrix: tf.Tensor,
    kernel_chol: tf.Tensor,
    metric: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    obs_noise_chol = tf.linalg.cholesky(model.observation_noise_cov)
    state_dim = model.state_dim

    def score_body(particle: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        particle = tf.convert_to_tensor(particle, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(particle)
            obs_pred = tf.convert_to_tensor(
                model.observation_fn(particle, control), dtype=tf.float32
            )
        obs_jacobian = tape.jacobian(obs_pred, particle)

        residual = observation - obs_pred
        score = tf.linalg.cholesky_solve(
            obs_noise_chol,
            residual[:, tf.newaxis],
        )
        score = tf.reshape(score, (-1,))

        score_jacobian = -tf.matmul(
            tf.transpose(obs_jacobian),
            tf.linalg.cholesky_solve(
                obs_noise_chol,
                tf.eye(model.observation_dim, dtype=tf.float32),
            ),
        )

        return score, score_jacobian

    raw_scores, score_jacobians = tf.map_fn(
        lambda x: score_body(x),
        particles,
        fn_output_signature=(
            tf.TensorSpec((state_dim,), tf.float32),
            tf.TensorSpec((state_dim, state_dim), tf.float32),
        ),
    )

    weights = tf.linalg.cholesky_solve(kernel_chol, raw_scores)
    preconditioned_scores = tf.matmul(kernel_matrix, weights)

    preconditioned_jacobians = tf.map_fn(
        lambda jac: tf.matmul(metric, jac),
        score_jacobians,
        fn_output_signature=tf.TensorSpec((state_dim, state_dim), tf.float32),
    )

    return preconditioned_scores, preconditioned_jacobians


def _sample_covariance(particles: tf.Tensor) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    mean = tf.reduce_mean(particles, axis=0, keepdims=True)
    centered = particles - mean
    num = tf.cast(tf.shape(particles)[0], tf.float32)
    denom = tf.maximum(num - 1.0, 1.0)
    cov = tf.matmul(centered, centered, transpose_a=True) / denom
    return cov


def _condition_number(matrix: tf.Tensor) -> tf.Tensor:
    singular_values = tf.linalg.svd(matrix, compute_uv=False)
    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]
    return sigma_max / tf.maximum(
        sigma_min,
        tf.constant(1e-6, dtype=tf.float32),
    )
