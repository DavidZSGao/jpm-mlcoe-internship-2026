import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from mlcoe_q1.pipelines import train_forecaster
from mlcoe_q1.utils.driver_features import BASE_FEATURES


def _sample_dataframe() -> pd.DataFrame:
    periods = pd.date_range("2020-03-31", periods=5, freq="QE-DEC")
    sales = np.linspace(100.0, 140.0, len(periods))
    data: dict[str, object] = {
        "ticker": ["AAA"] * len(periods),
        "period": periods,
        "sales": sales,
        "log_sales": np.log(sales),
        "sales_per_asset": np.linspace(0.6, 0.8, len(periods)),
        "sales_growth": np.linspace(0.02, 0.05, len(periods)),
        "ebit_margin": np.linspace(0.1, 0.15, len(periods)),
        "depreciation_ratio": np.linspace(0.02, 0.03, len(periods)),
        "capex_ratio": np.linspace(0.04, 0.05, len(periods)),
        "nwc_ratio": np.linspace(0.08, 0.1, len(periods)),
        "payout_ratio": np.linspace(0.2, 0.25, len(periods)),
        "leverage_ratio": np.linspace(0.3, 0.35, len(periods)),
    }
    return pd.DataFrame(data)


def test_parse_units_parses_comma_list() -> None:
    assert train_forecaster._parse_units("32, 16") == [32, 16]


def test_parse_units_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        train_forecaster._parse_units("0")
    with pytest.raises(ValueError):
        train_forecaster._parse_units("abc")


def test_build_sequence_dataset_shapes() -> None:
    df = _sample_dataframe()
    sequences, aux, targets = train_forecaster._build_sequence_dataset(
        df, BASE_FEATURES, sequence_length=3
    )
    assert sequences.shape == (len(df) - 3, 3, len(BASE_FEATURES))
    assert aux.shape == (len(df) - 3, len(train_forecaster.AUX_FEATURES))
    assert targets.shape == (len(df) - 3, len(BASE_FEATURES))


def test_build_dataset_empty_when_insufficient_history() -> None:
    df = _sample_dataframe().head(2)
    sequences, aux, targets = train_forecaster._build_sequence_dataset(
        df, BASE_FEATURES, sequence_length=3
    )
    assert sequences.size == 0
    assert aux.size == 0
    assert targets.size == 0


def test_variational_loss_reduces_to_gaussian_when_weight_zero() -> None:
    y_true = tf.constant([[0.1, -0.2]], dtype=tf.float32)
    mean = tf.constant([[0.1, -0.25]], dtype=tf.float32)
    log_var = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    y_pred = tf.concat([mean, log_var], axis=-1)

    gaussian = train_forecaster._gaussian_nll(y_true, y_pred)
    variational = train_forecaster.make_variational_loss(0.0)(y_true, y_pred)
    np.testing.assert_allclose(gaussian.numpy(), variational.numpy())
