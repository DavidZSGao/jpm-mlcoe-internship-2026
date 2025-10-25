
"""Validation utilities for the LGSSM Kalman filter implementation."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from mlcoe_q2.data.lgssm import LinearGaussianSSM
from mlcoe_q2.models.filters.kalman import kalman_filter
from mlcoe_q2.utils import add_config_argument, parse_args_with_config


@dataclasses.dataclass
class ValidationSummary:
    """Summary statistics for Kalman filter validation."""

    mean_abs_error_means: float
    max_abs_error_means: float
    rmse_means: float
    mean_abs_error_covs: float
    max_abs_error_covs: float
    joseph_vs_naive_diff: float
    transition_cond: float
    observation_cond: float


@dataclasses.dataclass
class ValidationArtifacts:
    """Supporting arrays for visualization and reporting."""

    states: np.ndarray
    observations: np.ndarray
    filtered_means: np.ndarray
    filtered_covs: np.ndarray
    reference_means: np.ndarray
    reference_covs: np.ndarray


def _reference_kalman(
    model: LinearGaussianSSM,
    observations: tf.Tensor,
    initial_mean: tf.Tensor,
    initial_cov: tf.Tensor,
    controls: Optional[tf.Tensor] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reference Kalman recursion using NumPy double precision."""

    obs = observations.numpy().astype(np.float64)
    mean = initial_mean.numpy().astype(np.float64)
    cov = initial_cov.numpy().astype(np.float64)

    A = model.transition_matrix.numpy().astype(np.float64)
    C = model.observation_matrix.numpy().astype(np.float64)
    Q = model.transition_cov.numpy().astype(np.float64)
    R = model.observation_cov.numpy().astype(np.float64)

    if model.control_matrix is not None:
        B = model.control_matrix.numpy().astype(np.float64)
    else:
        B = None

    if model.observation_control_matrix is not None:
        D = model.observation_control_matrix.numpy().astype(np.float64)
    else:
        D = None

    if controls is not None:
        U = controls.numpy().astype(np.float64)
    else:
        U = None

    identity = np.eye(A.shape[0], dtype=np.float64)

    filtered_means = []
    filtered_covs = []

    for t, obs_t in enumerate(obs):
        control_t = U[t] if U is not None else None

        # Prediction
        mean_pred = A @ mean
        if control_t is not None and B is not None:
            mean_pred = mean_pred + B @ control_t
        cov_pred = A @ cov @ A.T + Q

        # Update
        expected_obs = C @ mean_pred
        if control_t is not None and D is not None:
            expected_obs = expected_obs + D @ control_t
        innovation = obs_t - expected_obs
        innovation_cov = C @ cov_pred @ C.T + R
        gain = cov_pred @ C.T @ np.linalg.inv(innovation_cov)

        mean = mean_pred + gain @ innovation

        joseph_factor = identity - gain @ C
        cov = (
            joseph_factor @ cov_pred @ joseph_factor.T
            + gain @ R @ gain.T
        )

        filtered_means.append(mean.copy())
        filtered_covs.append(cov.copy())

    return np.array(filtered_means), np.array(filtered_covs)


def _simulate_lgssm(
    state_dim: int,
    observation_dim: int,
    num_steps: int,
    seed: int,
) -> tuple[
    LinearGaussianSSM,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
]:
    """Construct and simulate a moderately conditioned LGSSM."""

    tf.random.set_seed(seed)

    transition = tf.constant(
        [[0.9, 0.3, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 0.7]],
        dtype=tf.float32,
    )
    observation = tf.constant(
        [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0]],
        dtype=tf.float32,
    )
    transition_cov = tf.constant(
        [[0.05, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.02]],
        dtype=tf.float32,
    )
    observation_cov = tf.constant(
        [[0.04, 0.0], [0.0, 0.02]],
        dtype=tf.float32,
    )

    model = LinearGaussianSSM(
        transition_matrix=transition,
        observation_matrix=observation,
        transition_cov=transition_cov,
        observation_cov=observation_cov,
    )

    initial_state = tf.constant([0.1, -0.2, 0.05], dtype=tf.float32)
    states, observations = model.simulate(num_steps, initial_state)

    initial_mean = tf.zeros((state_dim,), dtype=tf.float32)
    initial_cov = tf.eye(state_dim, dtype=tf.float32)

    return model, states, observations, initial_mean, initial_cov


def validate_kalman_filter(
    num_steps: int = 75,
    seed: int = 7,
) -> tuple[ValidationSummary, ValidationArtifacts]:
    """Run validation comparing implementation against double precision reference."""

    (
        model,
        states,
        observations,
        init_mean,
        init_cov,
    ) = _simulate_lgssm(
        state_dim=3,
        observation_dim=2,
        num_steps=num_steps,
        seed=seed,
    )

    result = kalman_filter(
        model=model,
        observations=observations,
        initial_mean=init_mean,
        initial_cov=init_cov,
    )

    ref_means, ref_covs = _reference_kalman(
        model=model,
        observations=observations,
        initial_mean=init_mean,
        initial_cov=init_cov,
    )

    means = result.filtered_means.numpy()
    covs = result.filtered_covs.numpy()

    mean_diff = means - ref_means
    cov_diff = covs - ref_covs
    mean_abs_error_means = float(np.mean(np.abs(mean_diff)))
    max_abs_error_means = float(np.max(np.abs(mean_diff)))
    rmse_means = float(np.sqrt(np.mean(mean_diff**2)))

    mean_abs_error_covs = float(np.mean(np.abs(cov_diff)))
    max_abs_error_covs = float(np.max(np.abs(cov_diff)))

    # Compare Joseph stabilization to naive covariance update for symmetry preservation.
    naive_covs = []
    identity = tf.eye(model.state_dim, dtype=tf.float32)
    cov_pred = init_cov
    mean_pred = init_mean
    for t in range(num_steps):
        obs_t = observations[t]
        mean_pred = tf.linalg.matvec(model.transition_matrix, mean_pred)
        cov_pred = (
            model.transition_matrix
            @ cov_pred
            @ tf.transpose(model.transition_matrix)
            + model.transition_cov
        )
        innovation_cov = (
            model.observation_matrix
            @ cov_pred
            @ tf.transpose(model.observation_matrix)
            + model.observation_cov
        )
        gain = cov_pred @ tf.transpose(model.observation_matrix) @ tf.linalg.inv(
            innovation_cov
        )
        innovation = obs_t - tf.linalg.matvec(
            model.observation_matrix, mean_pred
        )
        mean_pred = mean_pred + tf.linalg.matvec(gain, innovation)
        cov_pred = (
            (identity - gain @ model.observation_matrix)
            @ cov_pred
        )
        naive_covs.append(cov_pred.numpy())
    naive_covs = np.array(naive_covs)

    joseph_covs = covs
    joseph_vs_naive = float(np.max(np.abs(joseph_covs - naive_covs)))

    transition_cond = float(
        np.linalg.cond(model.transition_cov.numpy())
    )
    observation_cond = float(
        np.linalg.cond(model.observation_cov.numpy())
    )

    summary = ValidationSummary(
        mean_abs_error_means=mean_abs_error_means,
        max_abs_error_means=max_abs_error_means,
        rmse_means=rmse_means,
        mean_abs_error_covs=mean_abs_error_covs,
        max_abs_error_covs=max_abs_error_covs,
        joseph_vs_naive_diff=joseph_vs_naive,
        transition_cond=transition_cond,
        observation_cond=observation_cond,
    )

    artifacts = ValidationArtifacts(
        states=states.numpy(),
        observations=observations.numpy(),
        filtered_means=means,
        filtered_covs=covs,
        reference_means=ref_means,
        reference_covs=ref_covs,
    )

    return summary, artifacts


def _format_summary(summary: ValidationSummary) -> str:
    lines = [
        "Kalman Filter Validation Summary",
        f"Mean abs error (means): {summary.mean_abs_error_means:.3e}",
        f"Max abs error (means): {summary.max_abs_error_means:.3e}",
        f"RMSE (means): {summary.rmse_means:.3e}",
        f"Mean abs error (covariances): {summary.mean_abs_error_covs:.3e}",
        f"Max abs error (covariances): {summary.max_abs_error_covs:.3e}",
        f"Joseph vs naive covariance diff (max abs): {summary.joseph_vs_naive_diff:.3e}",
        f"Condition number (transition_cov): {summary.transition_cond:.3e}",
        f"Condition number (observation_cov): {summary.observation_cond:.3e}",
    ]
    return "\n".join(lines)


def _artifacts_metrics(artifacts: ValidationArtifacts) -> dict[str, list[float]]:
    per_dim_mae = np.mean(
        np.abs(artifacts.filtered_means - artifacts.reference_means),
        axis=0,
    )
    per_dim_rmse = np.sqrt(
        np.mean(
            (artifacts.filtered_means - artifacts.reference_means) ** 2,
            axis=0,
        )
    )
    return {
        "per_dim_mae": per_dim_mae.astype(float).tolist(),
        "per_dim_rmse": per_dim_rmse.astype(float).tolist(),
    }


def _save_json(
    path: Optional[Path],
    summary: ValidationSummary,
    artifacts: ValidationArtifacts,
) -> None:
    if path is None:
        return
    payload = {
        "summary": dataclasses.asdict(summary),
        "metrics": _artifacts_metrics(artifacts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _plot_filtered_vs_reference(
    artifacts: ValidationArtifacts,
    figure_path: Optional[Path],
) -> None:
    if figure_path is None:
        return

    steps = np.arange(artifacts.filtered_means.shape[0])
    num_dims = artifacts.filtered_means.shape[1]

    fig, axes = plt.subplots(
        num_dims,
        1,
        figsize=(7, 2.5 * num_dims),
        sharex=True,
        constrained_layout=True,
    )
    if num_dims == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.plot(
            steps,
            artifacts.filtered_means[:, dim],
            label="TensorFlow KF",
            linewidth=1.6,
        )
        ax.plot(
            steps,
            artifacts.reference_means[:, dim],
            linestyle="--",
            label="Reference recursion",
            linewidth=1.4,
        )
        ax.plot(
            steps,
            artifacts.states[:, dim],
            linestyle=":",
            label="True state",
            linewidth=1.2,
        )
        ax.set_ylabel(f"State {dim}")
        ax.grid(True, linewidth=0.3, alpha=0.4)

    axes[-1].set_xlabel("Time step")
    axes[0].legend(loc="upper right")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the LGSSM Kalman filter implementation.")
    parser.add_argument("--num-steps", type=int, default=75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--figure-path", type=Path, default=None)
    add_config_argument(parser)
    return parse_args_with_config(parser, argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    summary, artifacts = validate_kalman_filter(
        num_steps=args.num_steps,
        seed=args.seed,
    )

    print(_format_summary(summary))
    _save_json(args.output_json, summary, artifacts)
    _plot_filtered_vs_reference(artifacts, args.figure_path)


if __name__ == "__main__":
    main()
