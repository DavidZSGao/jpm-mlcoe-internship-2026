"""Lightweight gradient-based sampler using differentiable particle filters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.models.filters.differentiable_pf import differentiable_particle_filter
from mlcoe_q2.models.resampling.neural_ot import NeuralOTResampler


@dataclass
class DPFHMCConfig:
    """Configuration for the gradient-informed sampler."""

    num_results: int = 40
    num_burnin_steps: int = 10
    step_size: float = 0.05
    prior_scale: float = 1.0
    num_particles: int = 32
    proposal_noise: float = 0.03


def run_dpf_hmc(
    build_model_fn: Callable[[tf.Tensor], NonlinearStateSpaceModel],
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    initial_theta: tf.Tensor,
    config: DPFHMCConfig,
    *,
    resampling_method: str = "ot_low",
    neural_resampler: NeuralOTResampler | None = None,
) -> dict[str, tf.Tensor]:
    """Run a simple gradient-informed sampler driven by differentiable PF."""

    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    initial_particles = tf.convert_to_tensor(initial_particles, dtype=tf.float32)
    theta = tf.Variable(initial_theta, dtype=tf.float32)

    total_steps = config.num_results + config.num_burnin_steps
    theta_dim = int(theta.shape[0])
    samples: list[np.ndarray] = []
    log_posts: list[float] = []
    accepts = 0

    def log_posterior(param: tf.Tensor) -> tf.Tensor:
        model = build_model_fn(param)
        pf_result = differentiable_particle_filter(
            model,
            observations,
            num_particles=config.num_particles,
            initial_particles=initial_particles,
            resampling_method=resampling_method,
            neural_resampler=neural_resampler,
        )
        prior_term = -0.5 * tf.reduce_sum(tf.square(param / config.prior_scale))
        return pf_result.log_likelihood + prior_term

    start = time.perf_counter()
    current_log_post = log_posterior(theta)

    for step in range(total_steps):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            log_post = log_posterior(theta)
        grad = tape.gradient(log_post, theta)
        if grad is None:
            grad = tf.zeros_like(theta)
        noise = tf.random.normal(tf.shape(theta), dtype=tf.float32)
        proposal = theta + config.step_size * grad + noise * config.proposal_noise
        proposal_log_post = log_posterior(proposal)
        log_accept_ratio = proposal_log_post - log_post
        if tf.math.log(tf.random.uniform((), minval=1e-6, maxval=1.0, dtype=tf.float32)) < log_accept_ratio:
            theta.assign(proposal)
            current_log_post = proposal_log_post
            accepts += 1
        else:
            current_log_post = log_post

        if step >= config.num_burnin_steps:
            samples.append(np.array(theta.numpy(), copy=True))
            log_posts.append(float(current_log_post.numpy()))

    runtime = time.perf_counter() - start
    if samples:
        samples_array = np.stack(samples, axis=0)
        ess = _effective_sample_size(samples_array)
    else:
        samples_array = np.zeros((0, theta_dim), dtype=np.float32)
        ess = np.zeros((theta_dim,), dtype=np.float32)
    acceptance_rate = accepts / max(total_steps, 1)

    return {
        "samples": tf.convert_to_tensor(samples_array, dtype=tf.float32),
        "log_posterior": tf.convert_to_tensor(log_posts, dtype=tf.float32),
        "acceptance_rate": tf.constant(acceptance_rate, dtype=tf.float32),
        "ess": tf.convert_to_tensor(ess, dtype=tf.float32),
        "runtime_seconds": tf.constant(runtime, dtype=tf.float32),
    }


def _effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """Approximate ESS for each dimension using positive autocorrelation sums."""

    if samples.size == 0:
        return np.zeros((samples.shape[1] if samples.ndim == 2 else 0,), dtype=np.float32)
    if samples.ndim != 2:
        raise ValueError("samples must have shape [num_samples, num_dims]")
    num_samples, num_dims = samples.shape
    ess = np.empty(num_dims, dtype=np.float32)
    for d in range(num_dims):
        chain = samples[:, d]
        chain = chain - np.mean(chain)
        var = np.var(chain)
        if var == 0.0:
            ess[d] = float(num_samples)
            continue
        autocorr = np.correlate(chain, chain, mode="full")[num_samples - 1 :] / (
            var * num_samples
        )
        positive_sum = 0.0
        for rho in autocorr[1:]:
            if rho <= 0:
                break
            positive_sum += rho
        denom = 1.0 + 2.0 * positive_sum
        ess[d] = float(num_samples / max(denom, 1e-6))
    return ess


__all__ = ["DPFHMCConfig", "run_dpf_hmc"]
