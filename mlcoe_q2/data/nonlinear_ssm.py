"""Utilities for nonlinear/non-Gaussian state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf

from mlcoe_q2.data.lgssm import NoiseSampler, default_random_normal

TransitionFn = Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]
ObservationFn = Callable[[tf.Tensor, Optional[tf.Tensor]], tf.Tensor]


@dataclass
class NonlinearStateSpaceModel:
    """General nonlinear state-space model (SSM).

    Attributes:
        state_dim: Dimension of the latent state space.
        observation_dim: Dimension of the observation space.
        transition_fn: Callable implementing the (possibly nonlinear) transition.
        observation_fn: Callable implementing the observation mapping.
        process_noise_cov: Covariance of process noise (positive definite).
        observation_noise_cov: Covariance of observation noise (positive definite).
        control_dim: Dimension of optional control vector (if any).
    """

    state_dim: int
    observation_dim: int
    transition_fn: TransitionFn
    observation_fn: ObservationFn
    process_noise_cov: tf.Tensor
    observation_noise_cov: tf.Tensor
    control_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.observation_dim <= 0:
            raise ValueError("state_dim and observation_dim must be positive")
        self.process_noise_cov = tf.convert_to_tensor(
            self.process_noise_cov, dtype=tf.float32
        )
        self.observation_noise_cov = tf.convert_to_tensor(
            self.observation_noise_cov, dtype=tf.float32
        )

    def simulate(
        self,
        num_timesteps: int,
        initial_state: tf.Tensor,
        process_noise_sampler: NoiseSampler = default_random_normal,
        observation_noise_sampler: NoiseSampler = default_random_normal,
        controls: Optional[tf.Tensor] = None,
        seed: Optional[int] = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Simulate latent states and observations."""
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        initial_state = tf.convert_to_tensor(initial_state, dtype=tf.float32)
        if initial_state.shape[-1] != self.state_dim:
            raise ValueError("initial_state dimension mismatch")
        if controls is not None:
            controls = tf.convert_to_tensor(controls, dtype=tf.float32)
            expected_shape = (num_timesteps, self.control_dim or 0)
            if controls.shape != expected_shape:
                raise ValueError(
                    f"controls must have shape {expected_shape}, got {controls.shape}"
                )

        states = tf.TensorArray(dtype=tf.float32, size=num_timesteps)
        observations = tf.TensorArray(dtype=tf.float32, size=num_timesteps)

        x_prev = initial_state
        for t in tf.range(num_timesteps):
            control_t = controls[t] if controls is not None else None
            x_t = self._sample_next_state(
                x_prev, control_t, process_noise_sampler, seed
            )
            y_t = self._sample_observation(
                x_t, control_t, observation_noise_sampler, seed
            )
            states = states.write(t, x_t)
            observations = observations.write(t, y_t)
            x_prev = x_t
        return states.stack(), observations.stack()

    def _sample_next_state(
        self,
        state: tf.Tensor,
        control: Optional[tf.Tensor],
        sampler: NoiseSampler,
        seed: Optional[int],
    ) -> tf.Tensor:
        mean = self.transition_fn(state, control)
        noise = sampler((self.state_dim,), seed)
        cov_chol = tf.linalg.cholesky(self.process_noise_cov)
        return mean + tf.linalg.matvec(cov_chol, noise)

    def _sample_observation(
        self,
        state: tf.Tensor,
        control: Optional[tf.Tensor],
        sampler: NoiseSampler,
        seed: Optional[int],
    ) -> tf.Tensor:
        mean = self.observation_fn(state, control)
        noise = sampler((self.observation_dim,), seed)
        cov_chol = tf.linalg.cholesky(self.observation_noise_cov)
        return mean + tf.linalg.matvec(cov_chol, noise)
