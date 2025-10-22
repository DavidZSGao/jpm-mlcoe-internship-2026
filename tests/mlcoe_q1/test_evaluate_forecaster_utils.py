import pytest

import numpy as np
import pytest

from mlcoe_q1.models.balance_sheet_constraints import BalanceSheetState
from mlcoe_q1.pipelines.evaluate_forecaster import (
    _decode_predictions,
    _parse_quantiles,
    _project_from_features,
)


def test_parse_quantiles_sorts_and_deduplicates():
    quantiles = _parse_quantiles("0.9, 0.1, 0.5, 0.5")
    assert quantiles == [0.1, 0.5, 0.9]


def test_parse_quantiles_accepts_sequence_and_nested_strings():
    quantiles = _parse_quantiles([0.9, "0.1", [0.5]])
    assert quantiles == [0.1, 0.5, 0.9]


def test_parse_quantiles_parses_json_like_strings():
    quantiles = _parse_quantiles("[0.2, 0.8]")
    assert quantiles == [0.2, 0.8]


def test_parse_quantiles_rejects_out_of_bounds():
    with pytest.raises(ValueError):
        _parse_quantiles("-0.1")
    with pytest.raises(ValueError):
        _parse_quantiles("1.1")


def test_project_from_features_builds_projection():
    feature_columns = [
        'sales',
        'sales_growth',
        'ebit_margin',
        'depreciation_ratio',
        'capex_ratio',
        'nwc_ratio',
        'payout_ratio',
        'leverage_ratio',
    ]
    transform_config: dict[str, str] = {}
    state_prev = BalanceSheetState(
        cash=100.0,
        receivables=40.0,
        inventory=30.0,
        other_current_assets=20.0,
        net_pp_and_e=150.0,
        other_non_current_assets=60.0,
        accounts_payable=35.0,
        short_term_debt=15.0,
        accrued_expenses=10.0,
        long_term_debt=120.0,
        other_liabilities=25.0,
        equity=195.0,
    )

    feature_values = [
        120.0,  # sales
        0.05,   # growth
        0.12,   # ebit margin
        0.03,   # depreciation ratio
        0.04,   # capex ratio
        0.1,    # nwc ratio
        0.2,    # payout ratio
        0.4,    # leverage ratio
    ]

    result = _project_from_features(
        feature_values,
        feature_columns,
        transform_config,
        state_prev,
    )
    assert result is not None
    assert result.state.total_assets() > 0


def test_project_from_features_length_mismatch():
    state_prev = BalanceSheetState(
        cash=10,
        receivables=10,
        inventory=10,
        other_current_assets=10,
        net_pp_and_e=10,
        other_non_current_assets=10,
        accounts_payable=10,
        short_term_debt=10,
        accrued_expenses=10,
        long_term_debt=10,
        other_liabilities=10,
        equity=10,
    )
    feature_columns = ['sales']
    with pytest.raises(ValueError):
        _project_from_features(
            [1.0, 2.0],
            feature_columns,
            {},
            state_prev,
        )


def test_decode_predictions_gaussian():
    preds_scaled = np.asarray([[1.0, 2.0, 0.1, -0.2]], dtype=float)
    mean, log_var = _decode_predictions(preds_scaled, 'gaussian', 2)
    np.testing.assert_allclose(mean, [[1.0, 2.0]])
    np.testing.assert_allclose(log_var, [[0.1, -0.2]])


def test_decode_predictions_variational():
    preds_scaled = np.asarray([[0.5, -0.5, 0.2, 0.0]], dtype=float)
    mean, log_var = _decode_predictions(preds_scaled, 'variational', 2)
    np.testing.assert_allclose(mean, [[0.5, -0.5]])
    np.testing.assert_allclose(log_var, [[0.2, 0.0]])


def test_decode_predictions_invalid_dim():
    preds_scaled = np.ones((2, 3), dtype=float)
    with pytest.raises(ValueError):
        _decode_predictions(preds_scaled, 'gaussian', 3)
    with pytest.raises(ValueError):
        _decode_predictions(preds_scaled, 'deterministic', 2)
