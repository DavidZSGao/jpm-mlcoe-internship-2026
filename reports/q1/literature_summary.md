# Balance Sheet Projection Literature Notes

This note summarises the modelling guidance from the references cited in Question 1. The
focus is on extracting actionable ideas for deterministic balance-sheet simulations and for
connecting the accounting-identity approach to ML and simulation tooling.

## Vélez-Pareja (2007) — *Forecasting Financial Statements with No Plugs and No Circularity*
- Argues that spreadsheet "plugs" mask errors because they enforce the accounting identity by
  construction even when upstream drivers are inconsistent.
- Provides a worked example that ties every balance-sheet and income-statement line back to
  policy assumptions (turnover ratios, payout rules, debt schedules), ensuring that assets and
  liabilities reconcile without residuals.
- Recommends iterative driver refinement rather than post-hoc balancing, reinforcing the need
  for explicit state updates in our `project_forward` implementation.

**Implications for Q1**
- Use policy-driven links (DSO, DPO, inventory turns, payout ratios) to compute dependent
  working-capital accounts directly from revenues and cost forecasts.
- Maintain an explicit financing module so debt and equity adjustments fall out of cash-flow
  reconciliation instead of being forced by a plug.

## Vélez-Pareja (2009) — *Constructing Consistent Financial Planning Models for Valuation*
- Presents a simplified but comprehensive template covering balance sheet, income statement,
  and cash budget, emphasising that all statements derive from shared policy inputs.
- Highlights the separation between operating flows and financing flows, advocating for cash
  budgeting to mediate between accrued earnings and funding needs.
- Demonstrates valuation using Capital Cash Flow once the statements are internally
  consistent, providing a bridge from deterministic projections to valuation metrics.

**Implications for Q1**
- Structure pipeline inputs so operating assumptions (margins, turnover) feed both accounting
  statements and cash-flow reconciliation, preserving consistency for downstream valuation or
  ML targets.
- Capture financing policy parameters (target leverage, issuance rules) alongside operating
  drivers so the simulator can enforce them deterministically.

## Mejía-Pelaez & Vélez-Pareja (2011) — *Analytical Solution to the Circularity Problem in the DCF Valuation Framework*
- Derives closed-form expressions for equity value, leveraged cost of equity, and WACC that
  remove iterative circularity when debt tax shields depend on the valuation output.
- Shows equivalence between the analytical formulation, Adjusted Present Value (APV), and
  iterative spreadsheet solutions, validating that deterministic templates can remain
  circularity-free without losing accuracy.
- Emphasises that target leverage assumptions must align with the analytical solution to avoid
  inconsistent valuations.

**Implications for Q1**
- When training ML models on template outputs, expose tax-shield and leverage assumptions as
  explicit features to keep the learning problem well-posed.
- Provide analytical baselines in evaluation notebooks to validate that numerical solvers (or
  ML surrogates) reproduce the circularity-free valuations.

## Shahnazarian (2004) — *A Dynamic Microeconometric Simulation Model for Incorporated Businesses*
- Introduces CIMOD, a simulation framework that couples difference equations for stock
  variables (balance-sheet accounts) with statistically estimated behavioural equations for
  flows.
- Uses optimisation-derived relationships to estimate decision rules (investment, financing)
  and feeds them into a deterministic simulator, blending econometric estimates with accounting
  identities.
- Incorporates tax-system details directly into the state evolution, showcasing how policy
  parameters alter balance-sheet trajectories.

**Implications for Q1**
- Reinforces the need for a hybrid architecture where learned modules (driver forecaster) feed
  deterministic update rules rather than replacing them.
- Suggests modelling exogenous drivers `x(t)` as policy levers (tax rates, macro factors) and
  firm behavioural responses estimated from data, consistent with the prompt hint about
  simulation inputs.

## Samonas (2015) — *Financial Forecasting, Analysis and Modelling*
- Provides practitioner guidance on integrating financial statements, stressing scenario
  management and sensitivity analysis to test forecast robustness.
- Advocates for modular spreadsheet models with clearly defined input, calculation, and output
  blocks to prevent hidden circularities and to facilitate auditing.

**Implications for Q1**
- Mirror the spreadsheet discipline in code: isolate inputs (`mlcoe_q1/data/processed`),
  calculations (`project_forward`, ML modules), and outputs (reports, metrics) so simulation
  runs are reproducible.
- Extend the processed-data summariser to surface scenario metadata (assumptions, versioning)
  for auditability, aligning with the book’s emphasis on traceability.

## Next Steps
1. Annotate the deterministic specification with citations back to these notes, especially for
   working-capital and financing policies.
2. Expand the processed-data describer to emit scenario metadata required for Samonas-style
   audit trails.
3. Begin drafting the literature bridge for Part 2 drawing on the newer LLM
   benchmarking references (Alonso, 2024; Farr, 2025; Zhang, 2025) to anchor the
   evaluation design and reproducibility checklists.
