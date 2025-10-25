"""Sinkhorn-based optimal transport utilities for particle resampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf


@dataclass
class SinkhornResult:
    """Container for entropy-regularised transport outputs."""

    transport: tf.Tensor
    iterations: int


def pairwise_squared_distances(particles: tf.Tensor) -> tf.Tensor:
    """Compute pairwise squared Euclidean distances between particles."""

    particles = tf.convert_to_tensor(particles, dtype=tf.float32)
    diffs = particles[:, tf.newaxis, :] - particles[tf.newaxis, :, :]
    return tf.reduce_sum(tf.square(diffs), axis=-1)


def entropy_regularized_transport(
    source_weights: tf.Tensor,
    particles: tf.Tensor,
    epsilon: tf.Tensor,
    num_iters: int,
    tolerance: float,
    *,
    target_weights: Optional[tf.Tensor] = None,
) -> SinkhornResult:
    """Compute an entropy-regularised transport plan via Sinkhorn iterations."""

    source_weights = _normalize_weights(source_weights)
    num_particles = tf.shape(particles)[0]
    if target_weights is None:
        target_weights = tf.fill(
            (num_particles,),
            1.0 / tf.cast(num_particles, tf.float32),
        )
    else:
        target_weights = _normalize_weights(target_weights)

    epsilon = tf.convert_to_tensor(epsilon, dtype=tf.float32)
    particles = tf.convert_to_tensor(particles, dtype=tf.float32)

    cost_matrix = pairwise_squared_distances(particles)
    kernel = tf.exp(-cost_matrix / tf.maximum(epsilon, 1e-6))

    eps = tf.constant(1e-8, dtype=tf.float32)
    u = tf.ones_like(source_weights) / tf.cast(num_particles, tf.float32)
    v = tf.ones_like(target_weights) / tf.cast(num_particles, tf.float32)

    last_transport = None
    for i in range(max(num_iters, 1)):
        Kv = tf.matmul(kernel, v[:, tf.newaxis])
        u = source_weights / tf.maximum(tf.squeeze(Kv, axis=-1), eps)
        Ku = tf.matmul(tf.transpose(kernel), u[:, tf.newaxis])
        v = target_weights / tf.maximum(tf.squeeze(Ku, axis=-1), eps)

        diag_u = tf.linalg.diag(u)
        diag_v = tf.linalg.diag(v)
        transport = tf.matmul(diag_u, tf.matmul(kernel, diag_v))

        if tolerance > 0.0 and tf.executing_eagerly():
            row_error = tf.reduce_max(
                tf.abs(tf.reduce_sum(transport, axis=1) - source_weights)
            )
            col_error = tf.reduce_max(
                tf.abs(tf.reduce_sum(transport, axis=0) - target_weights)
            )
            if float(row_error.numpy()) < tolerance and float(col_error.numpy()) < tolerance:
                last_transport = transport
                return SinkhornResult(transport=_normalize_rows(transport), iterations=i + 1)
        last_transport = transport

    if last_transport is None:
        last_transport = tf.matmul(
            tf.linalg.diag(u), tf.matmul(kernel, tf.linalg.diag(v))
        )
    return SinkhornResult(
        transport=_normalize_rows(last_transport),
        iterations=max(num_iters, 1),
    )


def _normalize_rows(matrix: tf.Tensor) -> tf.Tensor:
    row_sums = tf.reduce_sum(matrix, axis=1, keepdims=True)
    return matrix / tf.maximum(row_sums, tf.constant(1e-8, dtype=matrix.dtype))


def _normalize_weights(weights: tf.Tensor) -> tf.Tensor:
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    total = tf.reduce_sum(weights)
    num = tf.cast(tf.shape(weights)[0], tf.float32)
    total = tf.where(total > 0.0, total, num)
    return tf.where(
        total > 0.0,
        weights / total,
        tf.fill(tf.shape(weights), 1.0 / num),
    )


__all__ = [
    "SinkhornResult",
    "entropy_regularized_transport",
    "pairwise_squared_distances",
]
