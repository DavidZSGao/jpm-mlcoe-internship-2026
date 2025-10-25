"""Neural state-space models built on gated recurrent transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel


@dataclass
class NeuralLSTMParameters:
    """Weights for a lightweight LSTM-style transition."""

    W_f: tf.Tensor
    U_f: tf.Tensor
    b_f: tf.Tensor
    W_i: tf.Tensor
    U_i: tf.Tensor
    b_i: tf.Tensor
    W_o: tf.Tensor
    U_o: tf.Tensor
    b_o: tf.Tensor
    W_g: tf.Tensor
    U_g: tf.Tensor
    b_g: tf.Tensor
    W_y: tf.Tensor
    b_y: tf.Tensor


class NeuralLSTMStateSpace:
    """Constructs an LSTM-inspired nonlinear state-space model."""

    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        control_dim: int = 0,
        *,
        seed: int = 0,
    ) -> None:
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.control_dim = control_dim
        self.state_dim = 2 * latent_dim

        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        recurrent_initializer = tf.keras.initializers.Orthogonal(seed=seed + 1)

        def dense(shape, init):
            return tf.Variable(init(shape), trainable=False, dtype=tf.float32)

        def bias(shape):
            return tf.Variable(tf.zeros(shape, dtype=tf.float32), trainable=False)

        input_dim = control_dim if control_dim > 0 else latent_dim

        self.params = NeuralLSTMParameters(
            W_f=dense((input_dim, latent_dim), initializer),
            U_f=dense((latent_dim, latent_dim), recurrent_initializer),
            b_f=bias((latent_dim,)),
            W_i=dense((input_dim, latent_dim), initializer),
            U_i=dense((latent_dim, latent_dim), recurrent_initializer),
            b_i=bias((latent_dim,)),
            W_o=dense((input_dim, latent_dim), initializer),
            U_o=dense((latent_dim, latent_dim), recurrent_initializer),
            b_o=bias((latent_dim,)),
            W_g=dense((input_dim, latent_dim), initializer),
            U_g=dense((latent_dim, latent_dim), recurrent_initializer),
            b_g=bias((latent_dim,)),
            W_y=dense((latent_dim, observation_dim), initializer),
            b_y=bias((observation_dim,)),
        )

    def build_model(
        self,
        process_scale: tf.Tensor | float,
        observation_scale: tf.Tensor | float,
    ) -> NonlinearStateSpaceModel:
        process_scale = tf.convert_to_tensor(process_scale, dtype=tf.float32)
        observation_scale = tf.convert_to_tensor(observation_scale, dtype=tf.float32)

        process_cov = tf.linalg.diag(
            tf.fill((self.state_dim,), tf.square(process_scale))
        )
        observation_cov = tf.linalg.diag(
            tf.fill((self.observation_dim,), tf.square(observation_scale))
        )

        def transition_fn(state: tf.Tensor, control: Optional[tf.Tensor]) -> tf.Tensor:
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            control_vec = (
                tf.convert_to_tensor(control, dtype=tf.float32)
                if control is not None
                else tf.zeros((self.control_dim,), dtype=tf.float32)
            )
            hidden, cell = tf.split(state, num_or_size_splits=2, axis=-1)
            input_vec = control_vec if self.control_dim > 0 else hidden

            def gate(W: tf.Tensor, U: tf.Tensor, b: tf.Tensor, activation) -> tf.Tensor:
                linear = tf.matmul(input_vec[tf.newaxis, :], W) + tf.matmul(
                    hidden[tf.newaxis, :], U
                ) + b
                return activation(linear)

            f = gate(self.params.W_f, self.params.U_f, self.params.b_f, tf.sigmoid)
            i = gate(self.params.W_i, self.params.U_i, self.params.b_i, tf.sigmoid)
            o = gate(self.params.W_o, self.params.U_o, self.params.b_o, tf.sigmoid)
            g = gate(self.params.W_g, self.params.U_g, self.params.b_g, tf.tanh)

            new_cell = tf.squeeze(f * cell[tf.newaxis, :] + i * g, axis=0)
            new_hidden = tf.squeeze(o * tf.tanh(new_cell)[tf.newaxis, :], axis=0)
            return tf.concat([new_hidden, new_cell], axis=-1)

        def observation_fn(state: tf.Tensor, control: Optional[tf.Tensor]) -> tf.Tensor:
            state = tf.convert_to_tensor(state, dtype=tf.float32)
            hidden, _ = tf.split(state, num_or_size_splits=2, axis=-1)
            return tf.matmul(hidden[tf.newaxis, :], self.params.W_y) + self.params.b_y

        return NonlinearStateSpaceModel(
            state_dim=self.state_dim,
            observation_dim=self.observation_dim,
            transition_fn=lambda s, u: transition_fn(s, u),
            observation_fn=lambda s, u: tf.squeeze(observation_fn(s, u), axis=0),
            process_noise_cov=process_cov,
            observation_noise_cov=observation_cov,
            control_dim=self.control_dim if self.control_dim > 0 else None,
        )

    def initial_state(self) -> tf.Tensor:
        return tf.zeros((self.state_dim,), dtype=tf.float32)

    def sample_initial_particles(
        self, num_particles: int, scale: float = 0.2
    ) -> tf.Tensor:
        return tf.random.normal(
            (num_particles, self.state_dim),
            mean=0.0,
            stddev=scale,
            dtype=tf.float32,
        )


__all__ = [
    "NeuralLSTMParameters",
    "NeuralLSTMStateSpace",
]
