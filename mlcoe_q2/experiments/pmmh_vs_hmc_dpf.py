"""Bonus experiment: PMMH baseline vs HMC using Differentiable PF.

This script runs a small demonstration comparing a Particle Marginal Metropolis–Hastings
(PMMH) sampler (using a standard particle filter likelihood estimate) with an
HMC sampler that leverages a differentiable particle filter (DPF) to obtain
gradients of the log joint.

Notes
- This is a CPU-friendly scaffold intended for feasibility checks, not for
  production MCMC efficiency.
- We parameterize a single scalar: the observation noise log-std (phi). The
  model uses exp(phi) as observation noise std; process noise is kept fixed.
- PMMH uses the standard PF (`mlcoe_q2.filters.particle_filter`) to estimate the
  marginal likelihood; HMC uses the DPF (`mlcoe_q2.filters.differentiable_pf`)
  to obtain gradients of an approximate log-likelihood.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from mlcoe_q2.datasets.nonlinear_ssm import NonlinearStateSpaceModel
from mlcoe_q2.filters import particle_filter, differentiable_particle_filter


tfd = tfp.distributions


@dataclass
class PMMHResults:
    accept_rate: float
    samples: list[float]


@dataclass
class HMCResults:
    step_size: float
    num_leapfrog_steps: int
    accept_rate: float
    samples: list[float]


def build_model(phi: tf.Tensor) -> NonlinearStateSpaceModel:
    """Construct the nonlinear SSM given log-std parameter phi for observation noise."""
    obs_std = tf.exp(phi)
    obs_var = obs_std ** 2

    process_cov = tf.constant([[0.05, 0.0], [0.0, 0.03]], dtype=tf.float32)
    observation_cov = tf.reshape(obs_var, (1, 1))

    @tf.function(reduce_retracing=True)
    def transition_fn(state: tf.Tensor, control: Optional[tf.Tensor]) -> tf.Tensor:
        del control
        x0, x1 = state[0], state[1]
        new_x0 = 0.85 * x0 + 0.25 * tf.math.sin(x1) + 0.05 * tf.math.sin(3.0 * x0)
        new_x1 = 0.90 * x1 + 0.20 * tf.math.tanh(x0)
        return tf.stack([new_x0, new_x1])

    @tf.function(reduce_retracing=True)
    def observation_fn(state: tf.Tensor, control: Optional[tf.Tensor]) -> tf.Tensor:
        del control
        return tf.stack([0.6 * tf.math.sin(state[0]) + 0.1 * state[1] + 0.05 * tf.math.sin(2.5 * state[0])])

    return NonlinearStateSpaceModel(
        state_dim=2,
        observation_dim=1,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        process_noise_cov=process_cov,
        observation_noise_cov=observation_cov,
    )


def simulate_data(num_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    tf.random.set_seed(seed)
    # Use a modest true observation noise std
    true_phi = tf.constant(-0.5, dtype=tf.float32)  # exp(-0.5) ~ 0.61
    model = build_model(true_phi)
    initial = tf.constant([0.6, -0.4], dtype=tf.float32)
    states, obs = model.simulate(num_steps, initial_state=initial, seed=seed)
    return states.numpy(), obs.numpy()


def estimate_loglik_pf(phi: tf.Tensor, observations: tf.Tensor, num_particles: int, seed: int) -> tf.Tensor:
    """Estimate marginal log-likelihood with a standard PF (not differentiable)."""
    tf.random.set_seed(seed)
    model = build_model(phi)
    init_particles = tf.random.normal((num_particles, model.state_dim), dtype=tf.float32)
    result = particle_filter(
        model=model,
        observations=observations,
        num_particles=num_particles,
        initial_particles=init_particles,
    )
    return result.log_likelihood  # shape ()


def estimate_loglik_dpf(phi: tf.Tensor, observations: tf.Tensor, num_particles: int, seed: int) -> tf.Tensor:
    """Approximate marginal log-likelihood with DPF (differentiable)."""
    tf.random.set_seed(seed)
    model = build_model(phi)
    init_particles = tf.random.normal((num_particles, model.state_dim), dtype=tf.float32)
    dpf = differentiable_particle_filter(
        model=model,
        observations=observations,
        num_particles=num_particles,
        initial_particles=init_particles,
        mix_with_uniform=0.15,
        ot_epsilon=0.25,
        ot_num_iters=20,
        sinkhorn_tolerance=1e-3,
        resampling_method="ot_low",
    )
    return dpf.log_likelihood


def run_pmmh(
    observations: tf.Tensor,
    num_particles: int,
    iters: int,
    init_phi: float,
    proposal_std: float,
    seed: int,
) -> PMMHResults:
    tf.random.set_seed(seed)
    phi = tf.constant(init_phi, dtype=tf.float32)
    prior = tfd.Normal(loc=0.0, scale=1.0)  # N(0,1) prior on phi

    samples = []
    accepts = 0
    current_ll = float(estimate_loglik_pf(phi, observations, num_particles, seed).numpy())
    current_post = current_ll + float(prior.log_prob(phi).numpy())

    for i in range(iters):
        proposal = tf.constant(np.random.normal(float(phi.numpy()), proposal_std), dtype=tf.float32)
        prop_ll = float(estimate_loglik_pf(proposal, observations, num_particles, seed + i + 1).numpy())
        prop_post = prop_ll + float(prior.log_prob(proposal).numpy())
        log_alpha = prop_post - current_post  # symmetric RW proposal
        if np.log(np.random.rand()) < log_alpha:
            phi = proposal
            current_ll = prop_ll
            current_post = prop_post
            accepts += 1
        samples.append(float(phi.numpy()))
    accept_rate = accepts / max(iters, 1)
    return PMMHResults(accept_rate=accept_rate, samples=samples)


def run_hmc(
    observations: tf.Tensor,
    num_particles: int,
    num_samples: int,
    step_size: float,
    num_leapfrog_steps: int,
    init_phi: float,
    seed: int,
) -> HMCResults:
    tf.random.set_seed(seed)

    def target_log_prob_fn(phi_var: tf.Tensor) -> tf.Tensor:
        # Prior
        prior_lp = tfd.Normal(0.0, 1.0).log_prob(phi_var)
        # DPF approximate log-likelihood (differentiable)
        ll = estimate_loglik_dpf(phi_var, observations, num_particles, seed)
        return prior_lp + ll

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
    )
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=max(num_samples // 2, 1),
        target_accept_prob=0.65,
    )

    @tf.function(autograph=False, reduce_retracing=True)
    def sample_chain():
        init = tf.convert_to_tensor(init_phi, dtype=tf.float32)
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_samples,
            current_state=init,
            kernel=adaptive_kernel,
            num_burnin_steps=max(num_samples // 4, 1),
            trace_fn=lambda _, kr: kr.inner_results.is_accepted,
            seed=seed,
        )
        return samples, tf.reduce_mean(tf.cast(kernel_results, tf.float32))

    samples, mean_accept = sample_chain()
    return HMCResults(
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        accept_rate=float(mean_accept.numpy()),
        samples=[float(v) for v in samples.numpy().tolist()],
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-particles", type=int, default=64)
    p.add_argument("--pmmh-iters", type=int, default=200)
    p.add_argument("--pmmh-proposal-std", type=float, default=0.1)
    p.add_argument("--hmc-samples", type=int, default=200)
    p.add_argument("--hmc-step-size", type=float, default=0.1)
    p.add_argument("--hmc-leapfrog", type=int, default=5)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Simulate data
    _, obs = simulate_data(args.num_steps, args.seed)
    obs_tf = tf.convert_to_tensor(obs, dtype=tf.float32)

    # Run PMMH and HMC
    pmmh = run_pmmh(
        observations=obs_tf,
        num_particles=args.num_particles,
        iters=args.pmmh_iters,
        init_phi=0.0,
        proposal_std=args.pmmh_proposal_std,
        seed=args.seed + 101,
    )
    hmc = run_hmc(
        observations=obs_tf,
        num_particles=args.num_particles,
        num_samples=args.hmc_samples,
        step_size=args.hmc_step_size,
        num_leapfrog_steps=args.hmc_leapfrog,
        init_phi=0.0,
        seed=args.seed + 202,
    )

    payload = {
        "settings": {
            "num_steps": args.num_steps,
            "num_particles": args.num_particles,
            "pmmh_iters": args.pmmh_iters,
            "hmc_samples": args.hmc_samples,
        },
        "pmmh": asdict(pmmh),
        "hmc": asdict(hmc),
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
