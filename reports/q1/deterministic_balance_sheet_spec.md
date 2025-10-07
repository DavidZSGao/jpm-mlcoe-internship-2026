# Deterministic Balance Sheet Projection Specification

This note summarises the identity-preserving projection backbone used in `mlcoe_q1` and provides the mathematical detail requested in Question 1.

## State Representation

We track the balance sheet at time \(t\) with the state vector
\[
\mathbf{s}_t = (\text{cash}_t, \text{recv}_t, \text{inv}_t, \text{oca}_t, \text{pp\&e}_t, \text{onca}_t, \text{ap}_t, \text{std}_t, \text{accr}_t, \text{ltd}_t, \text{oliab}_t, \text{equity}_t)
\]
which matches the `BalanceSheetState` dataclass implementation.【F:mlcoe_q1/models/balance_sheet_constraints.py†L18-L57】 Assets and liabilities totals are derived as
\[
\text{assets}_t = (\text{cash}_t + \text{recv}_t + \text{inv}_t + \text{oca}_t) + (\text{pp\&e}_t + \text{onca}_t)
\]
\[
\text{liab}_t = (\text{ap}_t + \text{std}_t + \text{accr}_t) + (\text{ltd}_t + \text{oliab}_t)
\]
and the accounting identity requires \(\text{assets}_t = \text{liab}_t + \text{equity}_t\) after every projection step.【F:mlcoe_q1/models/balance_sheet_constraints.py†L32-L49】【F:mlcoe_q1/models/balance_sheet_constraints.py†L129-L143】

## Driver Vector

Operational and financing assumptions enter through the driver vector
\[
\mathbf{d}_t = (\text{sales}_t, g_t, m_t, \tau_t, \text{dep}_t, c_t, w_t, p_t, \lambda_t)
\]
where `sales` is the prior-period sales level, \(g_t\) the growth rate, \(m_t\) the EBIT margin, \(\tau_t\) the tax rate, `dep` depreciation, \(c_t\) capex as a share of sales, \(w_t\) the target net working-capital ratio, \(p_t\) the dividend payout ratio, and \(\lambda_t\) the target leverage (debt fraction).【F:mlcoe_q1/models/balance_sheet_constraints.py†L59-L73】

## Projection Flow

Given \(\mathbf{s}_{t-1}\) and \(\mathbf{d}_t\), the projection proceeds as follows.

1. **Income statement anchors**
   \[
   \begin{aligned}
   \text{sales}_t &= \text{sales}_{t-1} (1 + g_t),\\
   \text{EBIT}_t &= \text{sales}_t m_t.
   \end{aligned}
   \]
2. **Working-capital policy**
   The net working capital (excluding cash and debt) is pegged at \(w_t \cdot \text{sales}_t\) with a fixed split among receivables, inventory, and payables:
   \[
   \begin{aligned}
   \text{recv}_t &= 0.4 w_t \text{sales}_t,\\
   \text{inv}_t &= 0.4 w_t \text{sales}_t,\\
   \text{ap}_t &= 0.2 w_t \text{sales}_t.
   \end{aligned}
   \]
   Changes in these accounts yield \(\Delta \text{NWC}_t\) for the cash-flow statement.【F:mlcoe_q1/models/balance_sheet_constraints.py†L93-L111】
3. **Capital expenditure and depreciation**
   Net PP&E evolves deterministically via
   \[
   \text{pp\&e}_t = \max(0, \text{pp\&e}_{t-1} + c_t \text{sales}_t - \text{dep}_t).
   \]
   Other current and non-current asset/liability buckets are carried forward unchanged in the current template.【F:mlcoe_q1/models/balance_sheet_constraints.py†L85-L105】
4. **Financing policy**
   Invested capital is defined as operating assets minus non-interest-bearing liabilities:
   \[
   IC_t = (\text{recv}_t + \text{inv}_t + \text{oca}_t + \text{pp\&e}_t + \text{onca}_t) - (\text{ap}_t + \text{accr}_t + \text{oliab}_t).
   \]
   Target debt is \(D_t^{\star} = \max(0, \lambda_t IC_t)\) and is split 20 %/80 % between short- and long-term debt to compute interest expense on the average balance.【F:mlcoe_q1/models/balance_sheet_constraints.py†L111-L135】
5. **Earnings, taxes, and payouts**
   \[
   \begin{aligned}
   \text{EBT}_t &= \text{EBIT}_t - r \cdot \tfrac{1}{2}(D_{t-1} + D_t^{\star}),\\
   \text{tax}_t &= \max(0, \text{EBT}_t) \tau_t,\\
   \text{NI}_t &= \text{EBT}_t - \text{tax}_t,\\
   \text{div}_t &= \max(0, p_t \text{NI}_t),\\
   \text{equity}_t &= \text{equity}_{t-1} + (\text{NI}_t - \text{div}_t).
   \end{aligned}
   \]
   The policy implicitly reinvests retained earnings to support asset growth.【F:mlcoe_q1/models/balance_sheet_constraints.py†L135-L155】
6. **Cash reconciliation**
   All non-cash assets and financing balances are now specified; cash is the plug that enforces the accounting identity:
   \[
   \text{cash}_t = \max\bigl(0, (\text{ap}_t + \text{accr}_t + \text{oliab}_t + D_t^{\star} + \text{equity}_t) - (\text{recv}_t + \text{inv}_t + \text{oca}_t + \text{pp\&e}_t + \text{onca}_t)\bigr).
   \]
   The resulting gap \(\Delta = \text{assets}_t - (\text{liab}_t + \text{equity}_t)\) is reported as `identity_gap` for diagnostics and should be numerically negligible.【F:mlcoe_q1/models/balance_sheet_constraints.py†L143-L168】
7. **Cash-flow statement**
   Operating, investing, and financing cash flows reconcile the change in cash and expose any mismatch with the planned financing policy, enabling backtests of capital structure assumptions.【F:mlcoe_q1/models/balance_sheet_constraints.py†L143-L168】

## Sequential Simulation

Projecting over a horizon applies `project_forward` repeatedly, carrying the state forward:
\[
\{\mathbf{s}_t\}_{t=1}^T = \text{project\_horizon}(\mathbf{s}_0, \{\mathbf{d}_1, \dots, \mathbf{d}_T\}).
\]
This deterministic core lets us frame stochastic simulations by treating the driver sequence \(\{\mathbf{d}_t\}\) as either outputs from a statistical model (e.g., the MLP forecaster) or samples from a scenario generator \(x(t)\) alluded to in the prompt. Exogenous variables might include macro growth, commodity prices, or bank-specific credit spreads; once sampled, they map to driver parameters that feed the projection engine.【F:mlcoe_q1/models/balance_sheet_constraints.py†L168-L182】

## Integration Hooks

The functions are pure Python but designed to sit beneath differentiable modules: gradients from a TensorFlow model that predicts drivers pass through the deterministic reconciliation because the accounting steps are algebraic. This satisfies the prompt requirement to combine learning with strict identity enforcement while leaving room for richer ML extensions.
