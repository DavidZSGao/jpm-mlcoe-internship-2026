"""Compare DPF-driven HMC with Particle Gibbs on a neural state-space model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import tensorflow as tf

from mlcoe_q2.models.inference import run_dpf_hmc, run_particle_gibbs
from mlcoe_q2.models.inference.dpf_hmc import DPFHMCConfig
from mlcoe_q2.models.inference.particle_gibbs import ParticleGibbsConfig
from mlcoe_q2.models.neural_ssm import NeuralLSTMStateSpace
from mlcoe_q2.models.resampling.neural_ot import NeuralOTResampler
from mlcoe_q2.utils import add_config_argument, ensure_output_paths, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-timesteps", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--observation-dim", type=int, default=2)
    parser.add_argument("--control-dim", type=int, default=0)
    parser.add_argument("--true-log-process-scale", type=float, default=-0.2)
    parser.add_argument("--true-log-observation-scale", type=float, default=-0.4)
    parser.add_argument("--dpf-num-particles", type=int, default=96)
    parser.add_argument("--pg-num-particles", type=int, default=128)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--hmc-num-results", type=int, default=300)
    parser.add_argument("--hmc-burnin", type=int, default=75)
    parser.add_argument("--hmc-step-size", type=float, default=0.04)
    parser.add_argument("--hmc-proposal-noise", type=float, default=0.03)
    parser.add_argument("--pg-iterations", type=int, default=400)
    parser.add_argument("--pg-step-size", type=float, default=0.12)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument(
        "--neural-ot-model",
        type=Path,
        default=None,
        help="Optional path to a trained neural OT accelerator",
    )
    parser.add_argument(
        "--dpf-resampling",
        type=str,
        default="soft",
        choices=["soft", "ot_low", "ot", "neural_ot"],
        help="Resampling strategy for the differentiable particle filter",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/artifacts/neural_state_space_inference.json"),
    )
    parser.add_argument(
        "--status-md",
        type=Path,
        default=Path("reports/q2/status/bonus_neural_ssm.md"),
    )
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


def _build_model_fn(builder: NeuralLSTMStateSpace) -> Callable[[tf.Tensor], tf.Tensor]:
    def build(theta: tf.Tensor) -> tf.Tensor:
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        log_process, log_obs = tf.unstack(theta)
        process_scale = tf.exp(log_process)
        observation_scale = tf.exp(log_obs)
        return builder.build_model(process_scale, observation_scale)

    return build


def _summarise_chain(chain: np.ndarray, true_theta: np.ndarray) -> dict[str, float]:
    mean = chain.mean(axis=0)
    std = chain.std(axis=0)
    rmse = np.sqrt(np.mean((chain - true_theta) ** 2, axis=0))
    return {
        "mean_log_process": float(mean[0]),
        "mean_log_observation": float(mean[1]),
        "std_log_process": float(std[0]),
        "std_log_observation": float(std[1]),
        "rmse_log_process": float(rmse[0]),
        "rmse_log_observation": float(rmse[1]),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Bonus — Neural State-Space Inference",
        "",
        "| Method | Acceptance | ESS (log proc) | ESS (log obs) | Runtime (s) | RMSE (log proc) | RMSE (log obs) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| DPF-HMC | {summary['hmc']['acceptance_rate']:.2f} | {summary['hmc']['ess'][0]:.1f} | {summary['hmc']['ess'][1]:.1f} | {summary['hmc']['runtime_seconds']:.2f} | {summary['hmc']['rmse_log_process']:.3f} | {summary['hmc']['rmse_log_observation']:.3f} |",
        f"| Particle Gibbs | {summary['pg']['acceptance_rate']:.2f} | {summary['pg']['ess'][0]:.1f} | {summary['pg']['ess'][1]:.1f} | {summary['pg']['runtime_seconds']:.2f} | {summary['pg']['rmse_log_process']:.3f} | {summary['pg']['rmse_log_observation']:.3f} |",
        "",
        "## Posterior Means",
        "",
        f"- DPF-HMC: log process = {summary['hmc']['mean_log_process']:.3f}, log observation = {summary['hmc']['mean_log_observation']:.3f}",
        f"- Particle Gibbs: log process = {summary['pg']['mean_log_process']:.3f}, log observation = {summary['pg']['mean_log_observation']:.3f}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m mlcoe_q2.pipelines.neural_state_space_inference --config configs/q2/neural_state_space_inference.json",
        "```",
    ]
    path.write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tf.random.set_seed(args.random_seed)
    builder = NeuralLSTMStateSpace(
        latent_dim=args.latent_dim,
        observation_dim=args.observation_dim,
        control_dim=args.control_dim,
        seed=args.random_seed,
    )

    true_theta = np.array([
        args.true_log_process_scale,
        args.true_log_observation_scale,
    ], dtype=np.float32)

    true_model = builder.build_model(
        process_scale=tf.exp(tf.constant(args.true_log_process_scale, dtype=tf.float32)),
        observation_scale=tf.exp(tf.constant(args.true_log_observation_scale, dtype=tf.float32)),
    )
    _, observations = true_model.simulate(
        num_timesteps=args.num_timesteps,
        initial_state=builder.initial_state(),
        seed=args.random_seed,
    )
    observations = tf.convert_to_tensor(observations, dtype=tf.float32)

    dpf_particles = builder.sample_initial_particles(args.dpf_num_particles)
    pg_particles = builder.sample_initial_particles(args.pg_num_particles)

    build_model_fn = _build_model_fn(builder)

    resampling_method = args.dpf_resampling
    neural_resampler = None
    if args.neural_ot_model is not None:
        neural_resampler = NeuralOTResampler.from_saved_model(
            args.neural_ot_model,
            num_particles=args.dpf_num_particles,
            state_dim=builder.state_dim,
            compute_reference_error=False,
        )
        resampling_method = "neural_ot"

    dpf_config = DPFHMCConfig(
        num_results=args.hmc_num_results,
        num_burnin_steps=args.hmc_burnin,
        step_size=args.hmc_step_size,
        proposal_noise=args.hmc_proposal_noise,
        prior_scale=args.prior_scale,
        num_particles=args.dpf_num_particles,
    )

    pg_config = ParticleGibbsConfig(
        num_iterations=args.pg_iterations,
        step_size=args.pg_step_size,
        prior_scale=args.prior_scale,
        num_particles=args.pg_num_particles,
    )

    ensure_output_paths([args.output_json, args.status_md])

    hmc_results = run_dpf_hmc(
        build_model_fn,
        observations,
        dpf_particles,
        initial_theta=tf.constant(true_theta),
        config=dpf_config,
        resampling_method=resampling_method,
        neural_resampler=neural_resampler,
    )
    pg_results = run_particle_gibbs(
        build_model_fn,
        observations,
        pg_particles,
        initial_theta=tf.constant(true_theta),
        config=pg_config,
    )

    hmc_samples = hmc_results["samples"].numpy()
    pg_samples = pg_results["samples"].numpy()

    hmc_summary = {
        **_summarise_chain(hmc_samples, true_theta),
        "acceptance_rate": float(hmc_results["acceptance_rate"].numpy()),
        "ess": [float(x) for x in hmc_results["ess"].numpy()],
        "runtime_seconds": float(hmc_results["runtime_seconds"].numpy()),
    }
    pg_summary = {
        **_summarise_chain(pg_samples, true_theta),
        "acceptance_rate": float(pg_results["acceptance_rate"].numpy()),
        "ess": [float(x) for x in pg_results["ess"].numpy()],
        "runtime_seconds": float(pg_results["runtime_seconds"].numpy()),
    }

    config_payload = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }

    payload = {
        "config": {
            "pipeline": config_payload,
            "dpf_hmc": asdict(dpf_config),
            "particle_gibbs": asdict(pg_config),
        },
        "true_parameters": {
            "log_process": float(true_theta[0]),
            "log_observation": float(true_theta[1]),
        },
        "dpf_hmc": {
            **hmc_summary,
            "samples": hmc_samples.tolist(),
        },
        "particle_gibbs": {
            **pg_summary,
            "samples": pg_samples.tolist(),
            "log_likelihood_trace": [float(x) for x in pg_results["log_likelihoods"].numpy()],
        },
    }

    summary = {
        "hmc": hmc_summary,
        "pg": pg_summary,
    }

    _write_json(args.output_json, payload)
    _write_markdown(args.status_md, summary)


if __name__ == "__main__":
    main()
