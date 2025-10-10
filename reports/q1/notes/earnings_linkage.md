# Earnings Linkage Analysis

This note records how the deterministic balance-sheet projection integrates earnings forecasts and what work remains to improve accuracy.

## Current Coverage
- The `project_forward` engine emits an income statement alongside each projected balance sheet, including net income, EBIT, interest expense, and dividend flows. 【F:mlcoe_q1/models/balance_sheet_constraints.py†L64-L128】
- `evaluate_forecaster` now persists predicted versus observed net income for every evaluated period, enabling mean absolute error (MAE) tracking beside assets and equity metrics. 【F:mlcoe_q1/pipelines/evaluate_forecaster.py†L27-L159】
- `summarize_forecaster_eval` and `report_forecaster_status` ingest the new net-income MAE columns so earnings accuracy appears in aggregated dashboards and status highlights. 【F:mlcoe_q1/pipelines/summarize_forecaster_eval.py†L39-L71】【F:mlcoe_q1/pipelines/report_forecaster_status.py†L40-L96】

## Findings
- Net income MAE is available for neural and persistence projections (which route through `project_forward`); proportional bank templates emit balance-sheet states only, so earnings comparison is skipped for those rows. 【F:mlcoe_q1/models/bank_template.py†L28-L83】【F:mlcoe_q1/pipelines/evaluate_forecaster.py†L120-L157】
- Extracted ground-truth income metrics rely on Yahoo Finance `netIncome`-family fields; the new `IncomeStatementMetrics` helper normalises label variants before evaluation. 【F:mlcoe_q1/utils/state_extractor.py†L16-L112】

## Next Steps
1. Add feature engineering for income-statement drivers (e.g., interest rate spreads, efficiency ratios) so the neural forecaster learns richer earnings dynamics for banks.
2. Extend templates or hybrid strategies with heuristic net income estimates to keep bank tickers represented in earnings MAE tables.
3. Incorporate rolling baselines (e.g., year-over-year persistence) to contextualise the neural model’s incremental performance on earnings forecasts.
4. Reflect earnings accuracy trends in the final Part 1 report and in decision-support narratives for lending scenarios.
