# Question 1 Response Summary

This memo captures how the current codebase delivers the revised Question 1 requirements for the Strategic Lending Division, linking each prompt component to the relevant pipelines, documentation, and artifacts.

## Strategic Lending Application
- The repository implements a lending-focused forecasting stack anchored around the deterministic balance-sheet engine and probabilistic TensorFlow forecasters described in the Question 1 README, giving analysts commands for data ingestion, training, evaluation, scenario packaging, macro overlays, and bundled deliverables.【F:mlcoe_q1/README.md†L5-L90】【F:mlcoe_q1/README.md†L90-L164】
- Workplan tracking keeps the bank-underwriting framing front and centre, mapping prompt bullets to implemented assets and maintaining governance priorities for future hand-offs.【F:reports/q1/q1_workplan.md†L1-L58】【F:reports/q1/q1_workplan.md†L59-L82】

## Part 1 — Balance-Sheet Modelling & Forecasting
- Deterministic identity management (`mlcoe_q1/models/balance_sheet_constraints.py`) underpins the driver-based TensorFlow forecasters, which now support MLP and GRU architectures plus Gaussian and variational heads for probabilistic outputs.【F:mlcoe_q1/README.md†L21-L90】【F:mlcoe_q1/pipelines/train_forecaster.py†L1-L120】
- Evaluation pipelines enforce accounting identities while providing Monte Carlo intervals, multi-horizon projections, calibration diagnostics, and reasonableness scoring, enabling Strategic Lending to quantify accuracy and coverage before deploying scenarios.【F:mlcoe_q1/pipelines/evaluate_forecaster.py†L1-L200】【F:mlcoe_q1/pipelines/analyze_forecaster_calibration.py†L1-L140】【F:mlcoe_q1/pipelines/assess_scenario_reasonableness.py†L1-L180】
- Scenario packaging and macro-conditioning utilities translate probabilistic forecasts into lender-ready baseline/downside/upside tables and configurable macro shocks, with documentation to guide underwriting simulations.【F:mlcoe_q1/pipelines/package_scenarios.py†L1-L180】【F:mlcoe_q1/pipelines/simulate_macro_conditions.py†L1-L200】【F:reports/q1/q1_workplan.md†L17-L40】

## Part 2 — LLM Benchmarking & Reporting
- Prompt dataset builders, adapter-agnostic LLM runners, and the benchmarking suite orchestrate multi-model, multi-seed experiments across HuggingFace and hosted APIs while preserving manifest metadata and seed-aggregated variance tables for reproducibility; the new summariser ranks MAE/MAPE/coverage metrics and packages Markdown briefs for governance follow-up.【F:mlcoe_q1/pipelines/run_llm_adapter.py†L1-L220】【F:mlcoe_q1/pipelines/benchmark_llm_suite.py†L1-L220】【F:mlcoe_q1/pipelines/summarize_llm_benchmarks.py†L1-L200】
- Evaluation, comparison, and recommendation pipelines join structured forecasts with LLM outputs to surface accuracy, coverage, and CFO/C-suite narratives, feeding the Strategic Lending briefing and lender package.【F:mlcoe_q1/pipelines/evaluate_llm_responses.py†L1-L160】【F:mlcoe_q1/pipelines/compare_llm_and_forecaster.py†L1-L160】【F:mlcoe_q1/pipelines/generate_lending_briefing.py†L1-L200】
- PDF ratio extraction now spans corporates and global banks with provenance metadata and regression tests, supporting automated document ingestion in the LLM workflow.【F:mlcoe_q1/pipelines/extract_pdf_ratios.py†L1-L260】【F:reports/q1/q1_workplan.md†L11-L32】

## Bonus Tracks — Credit Analytics & Loan Pricing
- Altman dataset builders and scoring pipelines transform Yahoo Finance or manual metrics into repeatable credit ratings with stored artifacts for issuers such as Evergrande.【F:mlcoe_q1/pipelines/build_credit_rating_dataset.py†L1-L200】【F:mlcoe_q1/pipelines/score_credit_rating.py†L1-L160】
- Risk-warning extraction scans structured text chunks for going-concern, liquidity, regulatory, and operational red flags, emitting parquet detail files and JSON summaries for Strategic Lending triage.【F:mlcoe_q1/pipelines/extract_risk_warnings.py†L1-L120】
- Loan pricing utilities combine packaged scenarios with Altman scores and configurable macro sensitivities to produce decomposed base rates, macro adjustments, and spreads, aligning scenario forecasts with underwriting economics.【F:mlcoe_q1/credit/loan_pricing.py†L1-L200】【F:mlcoe_q1/pipelines/price_loans.py†L1-L200】【F:configs/q1/loan_pricing_macro.json†L1-L9】

## Deliverable Packaging & Next Actions
- The lender package compiler inventories strategic artifacts—briefings, scenarios, macro overlays, calibration tables, credit analytics, and pricing outputs—and can copy them into a manifest-driven deliverable bundle, while the publishing CLI layers on executive-summary regeneration, an optional artifact audit, and a zipped export for underwriting hand-offs.【F:mlcoe_q1/pipelines/compile_lending_package.py†L1-L220】【F:mlcoe_q1/pipelines/publish_lending_submission.py†L1-L260】
- The executive-summary pipeline consolidates evaluation, scenario, macro, LLM, credit, pricing, and risk outputs into a single Markdown memo for Strategic Lending leadership reviews.【F:mlcoe_q1/pipelines/generate_executive_summary.py†L1-L260】
- Governance next steps focus on finalising the consolidated memo, documenting hosted-LLM reproducibility, and exploring macro-conditioned pricing adjustments as optional enhancements.【F:reports/q1/q1_workplan.md†L59-L82】【F:reports/q1/q1_workplan.md†L82-L96】
