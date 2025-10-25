"""Base interfaces and shared utilities for particle flow filters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import tensorflow as tf

from mlcoe_q2.data.nonlinear_ssm import NonlinearStateSpaceModel


@dataclass
class ParticleFlowResult:
    """Outputs produced by a particle flow filtering step."""

    propagated_particles: tf.Tensor
    propagated_weights: tf.Tensor
    log_jacobians: tf.Tensor
    diagnostics: dict[str, tf.Tensor]


class ParticleFlow(Protocol):
    """Protocol for deterministic particle flow transformations."""

    def __call__(
        self,
        model: NonlinearStateSpaceModel,
        particles: tf.Tensor,
        weights: tf.Tensor,
        observation: tf.Tensor,
        control: tf.Tensor | None = None,
    ) -> ParticleFlowResult:
        """Propagate particles and weights given a new observation."""

