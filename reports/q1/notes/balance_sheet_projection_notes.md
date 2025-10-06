# Balance-Sheet Projection Module Notes

- Module: `mlcoe_q1/models/balance_sheet_constraints.py`
- Inputs: `BalanceSheetState` (prior-period balances) and `DriverVector` (sales, margins, tax rate, capex ratio, working-capital ratio, payout policy, target leverage).
- Outputs: `ProjectionResult` containing the next-period balance sheet, income statement summary, cash-flow reconciliation, and diagnostic `financing_gap` (difference between policy-implied financing and the cash reconciliation).
- Accounting enforcement:
  - Assets are balanced via an explicit cash solve; `identity_gap` is numerically zero by construction.
  - Working-capital accounts follow fixed splits (40/40/20) but can be generalised.
  - Interest expense uses average debt with a placeholder 5% rate; replace with market-implied curves in future iterations.
- Usage: hook into TensorFlow by predicting driver vectors and calling `project_forward` inside the loss to penalise `financing_gap` or deviations from observed statements.
- Next enhancements: support more granular schedules (depreciation by asset class, multi-tranche debt), expose Jacobians for differentiable training, and integrate scenario variables (tax rate shocks, capex overrides).
