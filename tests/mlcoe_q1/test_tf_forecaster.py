"""Tests for the TensorFlow forecaster architecture."""

import numpy as np
import tensorflow as tf

from mlcoe_q1.models.tf_forecaster import DriverConfig, build_mlp_forecaster


def test_bank_head_routes_predictions() -> None:
    """Bank samples should flow through the specialised head."""

    config = DriverConfig(
        feature_dim=2,
        aux_dim=1,
        output_dim=1,
        hidden_units=[],
        dropout=0.0,
        bank_feature_index=0,
    )
    model = build_mlp_forecaster(config)

    corp_head = model.get_layer("corp_head")
    bank_head = model.get_layer("bank_head")

    corp_head.set_weights(
        [
            np.asarray([[1.0], [1.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        ]
    )
    bank_head.set_weights(
        [
            np.asarray([[1.0], [1.0], [2.0]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        ]
    )

    non_bank_input = np.asarray([[1.0, 2.0, 0.0]], dtype=np.float32)
    bank_input = np.asarray([[1.0, 2.0, 1.0]], dtype=np.float32)

    non_bank_pred = model(non_bank_input, training=False).numpy()
    bank_pred = model(bank_input, training=False).numpy()

    np.testing.assert_allclose(non_bank_pred, [[3.0]])
    np.testing.assert_allclose(bank_pred, [[5.0]])


def test_build_mlp_forecaster_without_aux_features() -> None:
    """The model should operate when no auxiliary features are supplied."""

    config = DriverConfig(
        feature_dim=3,
        aux_dim=0,
        output_dim=2,
        hidden_units=[4],
        dropout=0.0,
        bank_feature_index=None,
    )
    model = build_mlp_forecaster(config)

    inputs = tf.ones((5, 3), dtype=tf.float32)
    outputs = model(inputs)

    assert outputs.shape == (5, 2)
