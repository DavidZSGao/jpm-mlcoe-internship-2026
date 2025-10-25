"""Resampling utilities for differentiable particle filters."""

from mlcoe_q2.models.resampling.neural_ot import (
    NeuralOTResampler,
    build_neural_ot_model,
    generate_ot_training_data,
)
from mlcoe_q2.models.resampling.sinkhorn import (
    SinkhornResult,
    entropy_regularized_transport,
    pairwise_squared_distances,
)

__all__ = [
    "NeuralOTResampler",
    "SinkhornResult",
    "build_neural_ot_model",
    "entropy_regularized_transport",
    "generate_ot_training_data",
    "pairwise_squared_distances",
]
