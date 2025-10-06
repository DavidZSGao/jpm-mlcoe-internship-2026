"""TensorFlow modules for driver forecasting."""

from __future__ import annotations

import tensorflow as tf
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DriverConfig:
    input_dim: int
    hidden_units: List[int]
    dropout: float = 0.0


def build_mlp_forecaster(config: DriverConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(config.input_dim,))
    x = inputs
    for units in config.hidden_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        if config.dropout:
            x = tf.keras.layers.Dropout(config.dropout)(x)
    outputs = tf.keras.layers.Dense(8, activation=None)(x)
    model = tf.keras.Model(inputs, outputs)
    return model

