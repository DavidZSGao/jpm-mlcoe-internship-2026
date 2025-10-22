# Simulation Strategy for Deterministic Balance-Sheet Models

This companion note to the deterministic specification documents how we extend the accounting projector into a stochastic simula
tion engine. The prompt frames the model as `y(t+1) = f(x(t), y(t)) + n(t)`; the following sections describe each component.

## State and Drivers

- **State (`y(t)`):** Balance-sheet snapshot represented by `BalanceSheetState` (cash, receivables, inventory, PP&E, current/long-
  term debt, other liabilities, equity).
- **Drivers (`x(t)`):** Exogenous controls produced by the driver dataset builder. Core elements include sales growth, EBITDA mar-
  gins, net working capital ratio, capex ratio, payout ratio, leverage target, and sector/bank flags. Optional covariates add tan-
  gible equity ratios, interest spreads, net-interest margins, and year-over-year growth signals.

## Noise Process (`n(t)`)

- **Deterministic Core:** The baseline simulator is noise-free; identity reconciliation ensures assets equal liabilities plus eq-
  uity.
- **Stochastic Extensions:** Introduce additive Gaussian noise on driver outputs (e.g., sales growth, margin shocks) with covari-
  ance estimated from historical residuals. TensorFlow Probability layers can draw `n(t)` during rollout so Monte Carlo scenari-
  os respect driver uncertainty.
- **Bank-Specific Disturbances:** Apply correlated shocks to leverage ratios and interest spreads to capture rate sensitivity in
  bank portfolios.

## Exogenous Variables (`x(t)`)

1. **Macro Factors:** GDP growth, CPI, policy rates, and credit spreads sourced from FRED or ECB data. These feed regressors for
   sales growth and leverage appetite.
2. **Commodity/FX Drivers:** For cyclicals (GM, CAT, UNP) include oil, freight, or metals indices. For banks include yield curve
   slopes and deposit beta estimates.
3. **LLM Signals:** In Part 2, incorporate structured summaries from annual-report analysis (risk flags, qualitative guidance) as
   categorical embeddings influencing payout or leverage adjustments.

## Simulation Workflow

1. **Driver Scenario Generation:** Sample macro factor paths (e.g., via VAR) and map them to driver adjustments using regression
   coefficients learned from historical data.
2. **Projection:** For each sample path, feed drivers into `project_forward` (or the bank ensemble) to roll the balance sheet for
   the desired horizon while enforcing identities.
3. **Risk Metrics:** Aggregate scenario results into distributions of assets, equity, net income, and regulatory ratios. Compute
   quantiles for capital-at-risk style reporting.
4. **Feedback Loop:** Use scenario outcomes to update driver priors (e.g., tighten payout ratios if equity shortfalls appear in a
   large fraction of simulations).

## Implementation Roadmap

- Extend the driver dataset builder to ingest macro time series and align them with financial statement periods.
- Fit regression layers mapping macro features to driver perturbations; expose them through the simulation CLI.
- Wrap the deterministic projector in a TensorFlow Probability `JointDistributionSequential` so sampling logic co-exists with the
  core accounting identities.
- Emit scenario summaries alongside existing evaluation artifacts to support CFO reporting and credit risk analytics. _Status:_ `mlcoe_q1.pipelines.package_scenarios` reshapes Monte Carlo quantiles into baseline/downside/upside tables, `mlcoe_q1.pipelines.simulate_macro_conditions` layers macro factor shocks (consensus, mild downturn, rate shock by default) onto those projections with adjustment provenance, `mlcoe_q1.pipelines.analyze_forecaster_calibration` quantifies coverage bias for each group, and `mlcoe_q1.pipelines.compile_lending_package` bundles the resulting scenarios, macro overlays, and briefings into a deliverable set for Strategic Lending reviewers.

