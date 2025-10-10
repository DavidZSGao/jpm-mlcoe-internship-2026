"""TensorFlow modules for driver forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import tensorflow as tf


@dataclass
class DriverConfig:
    feature_dim: int
    aux_dim: int
    output_dim: int
    hidden_units: List[int]
    dropout: float = 0.0
    bank_feature_index: Optional[int] = None


@tf.keras.utils.register_keras_serializable(package="mlcoe_q1")
class SliceLayer(tf.keras.layers.Layer):
    """Select a contiguous slice of the last dimension."""

    def __init__(self, start: int, size: Optional[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.start = int(start)
        self.size = None if size is None else int(size)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        if self.size is None:
            return inputs[:, self.start :]
        return inputs[:, self.start : self.start + self.size]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"start": self.start, "size": self.size})
        return config


@tf.keras.utils.register_keras_serializable(package="mlcoe_q1")
class BankIndicator(tf.keras.layers.Layer):
    """Extract the bank flag and add an explicit channel dimension."""

    def __init__(self, index: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        return tf.expand_dims(inputs[:, self.index], axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"index": self.index})
        return config


@tf.keras.utils.register_keras_serializable(package="mlcoe_q1")
class Complement(tf.keras.layers.Layer):
    """Compute 1 - x for weighting the non-bank head."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:  # type: ignore[override]
        return 1.0 - inputs


def build_mlp_forecaster(config: DriverConfig) -> tf.keras.Model:
    """Create an MLP with an optional bank-specific output head.

    The network consumes concatenated `[features, aux]` inputs. If
    ``bank_feature_index`` is provided, the auxiliary slice at that index is
    treated as an ``is_bank`` indicator and used to blend a dedicated bank head
    with the generic corporate head. This lets the model share parameters while
    still specialising its final layer for banking statements.
    """

    feature_dim = int(config.feature_dim)
    aux_dim = int(config.aux_dim)
    bank_feature_index = config.bank_feature_index

    total_dim = feature_dim + aux_dim
    inputs = tf.keras.Input(shape=(total_dim,), name="driver_inputs")

    features = SliceLayer(start=0, size=feature_dim, name="feature_slice")(inputs)
    aux = SliceLayer(start=feature_dim, size=None, name="aux_slice")(inputs) if aux_dim else None

    x = features
    for idx, units in enumerate(config.hidden_units):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"hidden_{idx}")(x)
        if config.dropout:
            x = tf.keras.layers.Dropout(config.dropout, name=f"dropout_{idx}")(x)

    corp_head = tf.keras.layers.Dense(
        config.output_dim, activation=None, name="corp_head"
    )(x)

    if aux is not None and bank_feature_index is not None:
        if bank_feature_index >= aux_dim:
            raise ValueError(
                "bank_feature_index must be within the auxiliary feature dimension"
            )

        bank_inputs = tf.keras.layers.Concatenate(name="bank_head_inputs")(
            [x, aux]
        )
        bank_head = tf.keras.layers.Dense(
            config.output_dim, activation=None, name="bank_head"
        )(bank_inputs)

        is_bank = BankIndicator(index=bank_feature_index, name="is_bank_slice")(aux)
        non_bank_weight = Complement(name="non_bank_weight")(is_bank)

        outputs = tf.keras.layers.Add(name="sector_blend")(
            [corp_head * non_bank_weight, bank_head * is_bank]
        )
    else:
        outputs = corp_head

    model = tf.keras.Model(inputs, outputs)
    return model
