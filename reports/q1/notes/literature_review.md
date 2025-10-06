# Balance-Sheet Forecasting Literature Notes

_Last updated: 2025-10-04_

## Vélez-Pareja (2007) — Forecasting Financial Statements with No Plugs and No Circularity
- Core idea: derive forward projections for the three financial statements by explicitly modelling the balance-sheet identities instead of using ad hoc “plug” accounts (e.g., cash surplus). The approach treats assets, liabilities, and equity as endogenous outcomes of operating, investing, and financing policies.
- Methodology: expresses working-capital, PP&E, and financing accounts as functions of operational drivers (sales growth, payout policy, capital expenditure plans) and solves the resulting system of balance-sheet difference equations analytically. Interest expense is determined by forecast debt outstanding, removing circular references that normally arise when debt depends on interest and vice-versa.
- Practical takeaways for our build:
  - Define a minimal set of independent drivers (revenues, margin assumptions, capital structure policy) and derive all other accounts from double-entry rules.
  - Implement forecasting functions that return balance-sheet line items after ensuring `Assets = Liabilities + Equity` at each horizon—no residual plugs.
  - Maintain separable schedules for working capital, fixed assets, taxes, and financing; each schedule feeds both the income statement and balance sheet simultaneously.

## Vélez-Pareja (2009/2010) — Constructing Consistent Financial Planning Models for Valuation
- Extends the 2007 framework to long-horizon valuation models. Focuses on building spreadsheet (or programmatic) templates that keep the three statements mutually consistent whether the firm is levered or unlevered.
- Provides checklists for identifying implicit plugs (e.g., forcing cash to absorb mismatches), and prescribes using debt policy equations (target leverage, amortisation tables) plus equity distribution rules to close the system.
- Introduces algebraic expressions for free cash flow, debt balances, and interest deductions that remove circularity and make models auditable.
- Implementation guidance for us:
  - Formalise financing policy modules (e.g., target debt-to-value, sinking schedules) as deterministic state updates so that interest expense, principal, and cash balances are all solved simultaneously.
  - Encode validation hooks that compare projected statements with the derived cash-flow identity `FCF = NOPAT + Depreciation - Capex - ΔNWC` and reconcile to changes in net debt/equity.
  - Use symbolic/automatic differentiation (or linear solves) instead of iteration to detect and eliminate hidden circular references.

## Mejía-Peláez & Vélez-Pareja (2011) — Analytical Solution to the Circularity Problem in DCF Models
- Problem addressed: when debt interest depends on debt balance, but debt balance also depends on interest (through cash-flow availability), spreadsheets exhibit circular references solved via iteration or temporary plugs.
- Contribution: derives a closed-form solution for debt outstanding and interest expense using matrix algebra. Debt is represented as a linear function of operating cash flows and financing policy parameters; solving the linear system yields interest without iteration.
- Implication for our TensorFlow prototype:
  - Represent financing updates as linear constraints that can be solved once per time step (or embedded as differentiable layers) rather than using while-loops that depend on convergence.
  - Encourages structuring the training loss to penalise violations of these linear equations, enabling the ML model to learn consistent outcomes with the analytical baseline as a regulariser.

## Shahnazarian (2004) — Dynamic Microeconometric Simulation Model
- Builds a micro-simulation engine for incorporated businesses that jointly forecasts the three statements under changing tax regimes.
- The balance-sheet evolution is governed by a system of difference equations; flow variables (investment, financing choices) are estimated empirically with Tobit/probit regressions calibrated on firm-level data.
- Highlights:
  - Explicitly models tax code features (depreciation allowances, investment credits) inside the difference equations.
  - Uses Monte Carlo simulation on the estimated behavioural equations to produce scenario distributions for each balance-sheet account.
- Relevance for us:
  - Suggests a hybrid architecture—deterministic accounting identities plus stochastic modules for managerial decisions (capex, dividends, financing) learned from data.
  - Provides a blueprint for incorporating policy variables (tax rates, depreciation schedules) as exogenous inputs `x(t)` in the state update function `y(t+1) = f(x(t), y(t)) + n(t)` outlined in the assignment brief.

## Additional Observations & Modelling Implications
- **Driver Selection:** All references emphasise a parsimonious driver set. For our ML formulation, we should forecast drivers (sales, margins, capex ratios) and then deterministically map them to full statements via the identity-preserving layer.
- **Analytical vs. ML Fusion:** The analytical formulas (Vélez-Pareja, Mejía-Peláez) can serve as hard constraints or differentiable projection operations inside TensorFlow to prevent the network from violating accounting identities.
- **Tax and Policy Inputs:** Shahnazarian demonstrates the importance of embedding tax rules directly in the transitions. Our data schema should therefore tag each account with tax-sensitivity metadata to support policy simulations.
- **Scenario Generation:** Simulation techniques from Shahnazarian can guide how we produce stress scenarios or augment training data (e.g., sampling alternative policy settings and macro paths).

## Next Steps
1. Translate the analytical identities into Python modules (`mlcoe_q1/models/balance_sheet_constraints.py`) that output consistent statements given a driver vector.
2. Implement linear-system solvers for debt/interest circularity and expose them as reusable functions for both classical forecasts and neural layers.
3. Incorporate policy/tax parameters into the processed dataset schema so they can be fed to both deterministic and stochastic components of the model.
