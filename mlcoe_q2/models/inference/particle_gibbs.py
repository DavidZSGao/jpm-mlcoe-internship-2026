"""Particle Gibbs-style inference with bootstrap particle filters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp

from mlcoe_q2.models.filters.particle import particle_filter
from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel


tfd = tfp.distributions


@dataclass
class ParticleGibbsConfig:
    """Configuration for PF-driven Particle Gibbs."""

    num_iterations: int = 400
    step_size: float = 0.1
    prior_scale: float = 1.0
    num_particles: int = 128


def run_particle_gibbs(
    build_model_fn: Callable[[tf.Tensor], NonlinearStateSpaceModel],
    observations: tf.Tensor,
    initial_particles: tf.Tensor,
    initial_theta: tf.Tensor,
    config: ParticleGibbsConfig,
) -> dict[str, tf.Tensor]:
    """Run a lightweight Particle Gibbs sampler using bootstrap PF proposals."""

    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    initial_particles = tf.convert_to_tensor(initial_particles, dtype=tf.float32)
    theta = tf.convert_to_tensor(initial_theta, dtype=tf.float32)

    samples = tf.TensorArray(dtype=tf.float32, size=config.num_iterations)
    log_likelihoods = tf.TensorArray(dtype=tf.float32, size=config.num_iterations)

    def log_prior(params: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(tfd.Normal(0.0, config.prior_scale).log_prob(params))

    def pf_log_likelihood(params: tf.Tensor) -> tf.Tensor:
        model = build_model_fn(params)
        result = particle_filter(
            model,
            observations,
            num_particles=config.num_particles,
            initial_particles=initial_particles,
        )
        return result.log_likelihood

    current_log_like = pf_log_likelihood(theta)
    current_log_prior = log_prior(theta)

    accepts = 0
    start_time = time.perf_counter()
    for i in range(config.num_iterations):
        proposal = theta + tf.random.normal(tf.shape(theta), stddev=config.step_size)
        proposal_log_like = pf_log_likelihood(proposal)
        proposal_log_prior = log_prior(proposal)

        log_accept_ratio = (
            proposal_log_like
            + proposal_log_prior
            - current_log_like
            - current_log_prior
        )
        accept = tf.math.log(tf.random.uniform(())) < log_accept_ratio
        if bool(accept.numpy()):
            theta = proposal
            current_log_like = proposal_log_like
            current_log_prior = proposal_log_prior
            accepts += 1

        samples = samples.write(i, theta)
        log_likelihoods = log_likelihoods.write(i, current_log_like)

    runtime = time.perf_counter() - start_time

    chain = samples.stack()
    ess = tfp.mcmc.effective_sample_size(chain)

    return {
        "samples": chain,
        "log_likelihoods": log_likelihoods.stack(),
        "acceptance_rate": tf.constant(accepts / config.num_iterations, dtype=tf.float32),
        "ess": ess,
        "runtime_seconds": tf.constant(runtime, dtype=tf.float32),
    }


__all__ = ["ParticleGibbsConfig", "run_particle_gibbs"]
