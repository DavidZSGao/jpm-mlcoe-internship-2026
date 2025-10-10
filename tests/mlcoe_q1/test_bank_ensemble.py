import json

import numpy as np

from mlcoe_q1.models.bank_ensemble import (
    BankEnsembleWeights,
    deserialize_ensemble,
    fit_ensemble_weights,
    serialize_ensemble,
)
from mlcoe_q1.models.balance_sheet_constraints import BalanceSheetState, ProjectionResult


def make_result(state: BalanceSheetState) -> ProjectionResult:
    return ProjectionResult(state=state, income_statement={}, cash_flow_statement={}, identity_gap=0.0)


def test_bank_ensemble_combination_respects_identity():
    template_state = BalanceSheetState(
        cash=50,
        receivables=100,
        inventory=20,
        other_current_assets=30,
        net_pp_and_e=200,
        other_non_current_assets=100,
        accounts_payable=40,
        short_term_debt=30,
        accrued_expenses=10,
        long_term_debt=120,
        other_liabilities=20,
        equity=280,
    )
    mlp_state = BalanceSheetState(
        cash=60,
        receivables=110,
        inventory=25,
        other_current_assets=35,
        net_pp_and_e=210,
        other_non_current_assets=105,
        accounts_payable=45,
        short_term_debt=25,
        accrued_expenses=12,
        long_term_debt=115,
        other_liabilities=18,
        equity=330,
    )

    weights = BankEnsembleWeights(
        ticker="TEST",
        assets_template_weight=0.4,
        assets_mlp_weight=0.6,
        assets_bias=5.0,
        equity_template_weight=0.2,
        equity_mlp_weight=0.8,
        equity_bias=2.0,
    )

    result = weights.combine(make_result(template_state), make_result(mlp_state))

    total_assets = result.state.total_assets()
    expected_assets = 0.4 * template_state.total_assets() + 0.6 * mlp_state.total_assets() + 5.0
    assert np.isclose(total_assets, expected_assets)
    assert np.isclose(result.state.total_assets(), result.state.total_liabilities() + result.state.equity)


def test_fit_and_serialize_roundtrip():
    rng = np.random.default_rng(0)
    template = rng.uniform(100, 200, size=5)
    mlp = rng.uniform(90, 210, size=5)
    true_assets = 0.3 * template + 0.7 * mlp + 10
    true_equity = 0.6 * template + 0.4 * mlp + 5

    records = [
        {
            "template_assets": float(t),
            "mlp_assets": float(m),
            "true_assets": float(a),
            "template_equity": float(t),
            "mlp_equity": float(m),
            "true_equity": float(e),
        }
        for t, m, a, e in zip(template, mlp, true_assets, true_equity)
    ]

    weights = fit_ensemble_weights(records, "ABC")
    assert np.isclose(weights.assets_template_weight, 0.3, atol=1e-6)
    assert np.isclose(weights.assets_mlp_weight, 0.7, atol=1e-6)
    assert np.isclose(weights.assets_bias, 10.0, atol=1e-6)

    payload = serialize_ensemble([weights])
    roundtrip = deserialize_ensemble(json.loads(json.dumps(payload)))
    assert "ABC" in roundtrip
    recovered = roundtrip["ABC"]
    assert np.isclose(recovered.equity_template_weight, weights.equity_template_weight)
    assert np.isclose(recovered.assets_bias, weights.assets_bias)

