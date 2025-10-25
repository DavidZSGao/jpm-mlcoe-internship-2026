"""Neural approximations for optimal transport resampling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import tensorflow as tf

from mlcoe_q2.models.resampling.sinkhorn import (
    entropy_regularized_transport,
    pairwise_squared_distances,
)


@dataclass
class NeuralOTConfig:
    """Configuration for training a neural OT accelerator."""

    num_particles: int = 8
    state_dim: int = 4
    epsilon_range: tuple[float, float] = (0.05, 0.5)
    num_samples: int = 1024
    validation_split: float = 0.15
    test_split: float = 0.1
    batch_size: int = 64
    epochs: int = 40
    learning_rate: float = 1e-3
    hidden_units: Iterable[int] = (128, 128)
    sinkhorn_iters: int = 30
    sinkhorn_tolerance: float = 1e-4
    random_seed: int = 7


def generate_ot_training_data(
    config: NeuralOTConfig,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Generate training/validation/test splits for neural OT acceleration."""

    rng = np.random.default_rng(config.random_seed)
    num_features = config.num_particles ** 2 + config.num_particles + 1

    features = np.zeros((config.num_samples, num_features), dtype=np.float32)
    targets = np.zeros((config.num_samples, config.num_particles ** 2), dtype=np.float32)

    for idx in range(config.num_samples):
        particles = rng.normal(size=(config.num_particles, config.state_dim)).astype(np.float32)
        raw_weights = rng.gamma(shape=1.5, scale=1.0, size=(config.num_particles,)).astype(np.float32)
        weights = raw_weights / np.maximum(np.sum(raw_weights), 1e-6)
        epsilon = rng.uniform(*config.epsilon_range)

        result = entropy_regularized_transport(
            tf.convert_to_tensor(weights),
            tf.convert_to_tensor(particles),
            tf.convert_to_tensor(epsilon),
            config.sinkhorn_iters,
            config.sinkhorn_tolerance,
        )
        transport = tf.reshape(result.transport, (-1,)).numpy()
        cost_matrix = pairwise_squared_distances(particles).numpy()

        feature = np.concatenate(
            [weights, cost_matrix.reshape(-1), np.array([epsilon], dtype=np.float32)]
        )
        features[idx] = feature
        targets[idx] = transport

    val_size = int(config.validation_split * config.num_samples)
    test_size = int(config.test_split * config.num_samples)
    train_size = config.num_samples - val_size - test_size

    x_train = features[:train_size]
    y_train = targets[:train_size]
    x_val = features[train_size : train_size + val_size]
    y_val = targets[train_size : train_size + val_size]
    x_test = features[train_size + val_size :]
    y_test = targets[train_size + val_size :]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_neural_ot_model(config: NeuralOTConfig) -> tf.keras.Model:
    """Construct a dense network to approximate Sinkhorn transport plans."""

    num_features = config.num_particles ** 2 + config.num_particles + 1
    inputs = tf.keras.Input(shape=(num_features,), dtype=tf.float32)
    x = inputs
    for units in config.hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
    x = tf.keras.layers.Dense(config.num_particles ** 2)(x)
    outputs = tf.keras.layers.Activation("linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


class NeuralOTResampler:
    """Wrapper around a trained neural model for OT acceleration."""

    def __init__(
        self,
        model: tf.keras.Model,
        num_particles: int,
        state_dim: int,
        *,
        reference_iters: int = 30,
        reference_tolerance: float = 1e-4,
        compute_reference_error: bool = False,
    ) -> None:
        self.model = model
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.reference_iters = reference_iters
        self.reference_tolerance = reference_tolerance
        self.compute_reference_error = compute_reference_error

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_path = path if path.suffix else path.with_suffix(".keras")
        tf.keras.models.save_model(self.model, save_path)

    @classmethod
    def from_saved_model(
        cls,
        path: Path | str,
        num_particles: int,
        state_dim: int,
        *,
        reference_iters: int = 30,
        reference_tolerance: float = 1e-4,
        compute_reference_error: bool = False,
    ) -> "NeuralOTResampler":
        model_path = Path(path)
        if model_path.suffix == "" and model_path.with_suffix(".keras").exists():
            model_path = model_path.with_suffix(".keras")
        model = tf.keras.models.load_model(model_path)
        return cls(
            model,
            num_particles,
            state_dim,
            reference_iters=reference_iters,
            reference_tolerance=reference_tolerance,
            compute_reference_error=compute_reference_error,
        )

    def _feature_vector(
        self,
        weights: tf.Tensor,
        particles: tf.Tensor,
        epsilon: tf.Tensor,
    ) -> tf.Tensor:
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        particles = tf.convert_to_tensor(particles, dtype=tf.float32)
        epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
        cost = pairwise_squared_distances(particles)
        flat = tf.concat(
            [
                tf.reshape(weights, (-1,)),
                tf.reshape(cost, (-1,)),
                tf.reshape(epsilon, (1,)),
            ],
            axis=0,
        )
        return tf.reshape(flat, (1, -1))

    def predict_transport(
        self,
        weights: tf.Tensor,
        particles: tf.Tensor,
        epsilon: tf.Tensor,
    ) -> tuple[tf.Tensor, Optional[tf.Tensor]]:
        features = self._feature_vector(weights, particles, epsilon)
        raw = self.model(features, training=False)
        transport = tf.reshape(raw, (self.num_particles, self.num_particles))
        transport = tf.nn.relu(transport)
        row_sums = tf.reduce_sum(transport, axis=1, keepdims=True)
        uniform_row = tf.fill(
            tf.shape(transport), 1.0 / tf.cast(self.num_particles, tf.float32)
        )
        normalized = transport / tf.maximum(row_sums, 1e-8)
        transport = tf.where(row_sums > 1e-6, normalized, uniform_row)

        error = None
        if self.compute_reference_error and tf.executing_eagerly():
            reference = entropy_regularized_transport(
                weights,
                particles,
                epsilon,
                self.reference_iters,
                self.reference_tolerance,
            ).transport
            diff = transport - reference
            denom = tf.maximum(tf.linalg.norm(reference), tf.constant(1e-6, dtype=tf.float32))
            error = tf.linalg.norm(diff) / denom
        return transport, error

    def benchmark_runtime(
        self,
        batch: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, float]:
        features, _ = batch
        features = tf.convert_to_tensor(features, dtype=tf.float32)

        start = time.perf_counter()
        _ = self.model(features, training=False)
        neural_time = time.perf_counter() - start

        sinkhorn_start = time.perf_counter()
        for idx in range(features.shape[0]):
            weights = features[idx, : self.num_particles]
            cost_flat = features[idx, self.num_particles : -1]
            epsilon = features[idx, -1]
            particles = self._reconstruct_particles(cost_flat)
            entropy_regularized_transport(
                weights,
                particles,
                epsilon,
                self.reference_iters,
                self.reference_tolerance,
            )
        sinkhorn_time = time.perf_counter() - sinkhorn_start

        count = float(features.shape[0])
        return {
            "neural_seconds": neural_time,
            "sinkhorn_seconds": sinkhorn_time,
            "per_sample_neural": neural_time / count,
            "per_sample_sinkhorn": sinkhorn_time / count,
        }

    def _reconstruct_particles(self, cost_flat: tf.Tensor) -> tf.Tensor:
        cost_matrix = tf.reshape(cost_flat, (self.num_particles, self.num_particles))
        # Recover particles via classical MDS-style embedding using eigendecomposition.
        # This keeps benchmarking differentiable-free and consistent with generated data.
        centering = tf.eye(self.num_particles) - tf.fill(
            (self.num_particles, self.num_particles),
            1.0 / tf.cast(self.num_particles, tf.float32),
        )
        gram = -0.5 * tf.matmul(centering, tf.matmul(cost_matrix, centering))
        eigvals, eigvecs = tf.linalg.eigh(gram)
        positive = tf.maximum(eigvals, 0.0)
        coords = eigvecs[:, -self.state_dim :] * tf.sqrt(positive[-self.state_dim :])
        return coords


def train_neural_ot_accelerator(
    config: NeuralOTConfig,
    *,
    model_dir: Optional[Path | str] = None,
) -> dict[str, float]:
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = generate_ot_training_data(config)
    model = build_neural_ot_model(config)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    resampler = NeuralOTResampler(
        model,
        num_particles=config.num_particles,
        state_dim=config.state_dim,
        reference_iters=config.sinkhorn_iters,
        reference_tolerance=config.sinkhorn_tolerance,
        compute_reference_error=True,
    )

    runtime = resampler.benchmark_runtime((x_test, y_test))
    if runtime["per_sample_neural"] > 0.0:
        runtime["relative_speedup"] = runtime["per_sample_sinkhorn"] / runtime[
            "per_sample_neural"
        ]
    else:
        runtime["relative_speedup"] = float("nan")

    predictions = model.predict(x_test, verbose=0)
    predictions = predictions.reshape((-1, config.num_particles, config.num_particles))
    predictions = np.maximum(predictions, 0.0)
    row_sums = np.maximum(np.sum(predictions, axis=2, keepdims=True), 1e-6)
    predictions = predictions / row_sums
    targets = y_test.reshape((-1, config.num_particles, config.num_particles))
    row_error = np.mean(np.abs(np.sum(predictions, axis=2) - 1.0))
    plan_l1 = np.mean(np.abs(predictions - targets))

    if model_dir is not None:
        resampler.save(model_dir)

    return {
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "row_normalization_error": float(row_error),
        "plan_l1_error": float(plan_l1),
        **runtime,
    }


__all__ = [
    "NeuralOTConfig",
    "NeuralOTResampler",
    "build_neural_ot_model",
    "generate_ot_training_data",
    "train_neural_ot_accelerator",
]
