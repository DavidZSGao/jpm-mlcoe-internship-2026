"""Linear Gaussian state-space model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf


def default_random_normal(
    shape: tuple[int, ...], seed: Optional[int] = None
) -> tf.Tensor:
    """Utility to sample from a standard normal distribution."""
    return tf.random.normal(
        shape,
        mean=0.0,
        stddev=1.0,
        seed=seed,
        dtype=tf.float32,
    )


NoiseSampler = Callable[[tuple[int, ...], Optional[int]], tf.Tensor]


@dataclass
class LinearGaussianSSM:
    """Linear-Gaussian state-space model (LGSSM).

    The dynamics follow::

        x_{t+1} = A x_t + B u_t + w_t,   w_t ~ N(0, Q)
        y_t = C x_t + D u_t + v_t,       v_t ~ N(0, R)

    where ``x_t`` and ``y_t`` are latent states and observations, respectively.
    """

    transition_matrix: tf.Tensor
    observation_matrix: tf.Tensor
    transition_cov: tf.Tensor
    observation_cov: tf.Tensor
    control_matrix: Optional[tf.Tensor] = None
    observation_control_matrix: Optional[tf.Tensor] = None
    state_dim: Optional[int] = None
    observation_dim: Optional[int] = None

    def __post_init__(self) -> None:
        self.transition_matrix = tf.convert_to_tensor(
            self.transition_matrix, dtype=tf.float32
        )
        self.observation_matrix = tf.convert_to_tensor(
            self.observation_matrix, dtype=tf.float32
        )
        self.transition_cov = tf.convert_to_tensor(
            self.transition_cov, dtype=tf.float32
        )
        self.observation_cov = tf.convert_to_tensor(
            self.observation_cov, dtype=tf.float32
        )
        if self.control_matrix is not None:
            self.control_matrix = tf.convert_to_tensor(
                self.control_matrix,
                dtype=tf.float32,
            )
        if self.observation_control_matrix is not None:
            self.observation_control_matrix = tf.convert_to_tensor(
                self.observation_control_matrix,
                dtype=tf.float32,
            )
        self.state_dim = (
            int(self.transition_matrix.shape[0])
            if self.state_dim is None
            else self.state_dim
        )
        self.observation_dim = (
            int(self.observation_matrix.shape[0])
            if self.observation_dim is None
            else self.observation_dim
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
        """Simulate latent states and observations for the LGSSM."""
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        initial_state = tf.convert_to_tensor(initial_state, dtype=tf.float32)
        if initial_state.shape[-1] != self.state_dim:
            raise ValueError("initial_state dimension mismatch")
        if controls is not None:
            controls = tf.convert_to_tensor(controls, dtype=tf.float32)
            if self.control_matrix is None:
                raise ValueError(
                    "control inputs provided but control_matrix is None"
                )
            expected_shape = (
                num_timesteps,
                int(self.control_matrix.shape[-1]),
            )
            if controls.shape != expected_shape:
                raise ValueError(
                    "controls must have shape"
                    f" {expected_shape}, got {controls.shape}"
                )
        elif self.observation_control_matrix is not None:
            raise ValueError(
                "observation_control_matrix provided but no controls supplied"
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
        noise_sampler: NoiseSampler,
        seed: Optional[int],
    ) -> tf.Tensor:
        mean = tf.linalg.matvec(self.transition_matrix, state)
        if control is not None and self.control_matrix is not None:
            mean = mean + tf.linalg.matvec(
                self.control_matrix,
                control,
            )
        noise = noise_sampler((self.state_dim,), seed)
        cov_chol = tf.linalg.cholesky(self.transition_cov)
        return mean + tf.linalg.matvec(cov_chol, noise)

    def _sample_observation(
        self,
        state: tf.Tensor,
        control: Optional[tf.Tensor],
        noise_sampler: NoiseSampler,
        seed: Optional[int],
    ) -> tf.Tensor:
        mean = tf.linalg.matvec(self.observation_matrix, state)
        if control is not None and self.observation_control_matrix is not None:
            mean = mean + tf.linalg.matvec(
                self.observation_control_matrix,
                control,
            )
        noise = noise_sampler((self.observation_dim,), seed)
        cov_chol = tf.linalg.cholesky(self.observation_cov)
        return mean + tf.linalg.matvec(cov_chol, noise)
