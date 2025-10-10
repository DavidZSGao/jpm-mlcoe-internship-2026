# Question 1 Execution Plan

This document maps the internship prompt requirements to the current repository assets and highlights the next set of concrete deliverables.

## Prompt Decomposition

### Part 1 — Balance Sheet Modelling
- **Literature grounding:** Summarise the accounting identity approach from Vélez-Pareja (2007, 2009, 2010) and Mejía-Pelaez & Vélez-Pareja (2011); extract the minimal equation set required for deterministic balance-sheet projection respecting `Assets = Liabilities + Equity` and related relationships.【F:Intern interview 2026 question 1.txt†L19-L36】 _Status:_ captured in `reports/q1/literature_summary.md` with implementation takeaways.
- **Mathematical specification:** Formalise the evolution equations for balance-sheet fields with explicit handling of dependent accounts (e.g., working-capital tie-outs, debt schedules). Provide a note on framing the problem as a time series and the identity management strategy.【F:Intern interview 2026 question 1.txt†L26-L33】
- **TensorFlow implementation:** Translate the specification into trainable modules layered on top of the existing `BalanceSheetState` / `project_forward` machinery; ensure gradients propagate through constraint adjustments.【F:Intern interview 2026 question 1.txt†L31-L33】
- **Data acquisition:** Extend the `yfinance` ingestion CLI to cover the tickers required for experiments; cache raw JSON responses for reproducibility.【F:Intern interview 2026 question 1.txt†L33-L35】
- **Training and evaluation:** Define experiment splits, loss metrics, and backtests to validate the forecasts and accounting identity compliance.【F:Intern interview 2026 question 1.txt†L35-L42】
- **Earnings linkage:** Document whether the driver/constraint framework can forecast earnings and what additional modelling is necessary.【F:Intern interview 2026 question 1.txt†L42-L43】 _Status:_ `evaluate_forecaster` now logs predicted vs. actual net income, surfaced through the summariser/status CLIs and expanded in `reports/q1/notes/earnings_linkage.md` with follow-on tasks.
- **ML extensions:** Catalogue candidate techniques (seq2seq, normalising flows, probabilistic forecasting) suitable for improving the baseline.【F:Intern interview 2026 question 1.txt†L43-L44】
- **Simulation view:** Identify the exogenous variables `x(t)` to support stochastic simulations consistent with the hint in the prompt.【F:Intern interview 2026 question 1.txt†L44-L46】

#### Part 1 Completion Checklist
- [x] Deterministic balance-sheet projection spec and TensorFlow implementation wired into the driver-based training/evaluation pipelines.
- [x] Raw statement ingestion cached for the nine focus tickers with processed features, validation, and evaluation CLIs.
- [x] Earnings metrics surfaced through evaluation and status-report tooling with documentation in `reports/q1/notes/earnings_linkage.md`.
- [x] Bank calibration: reduce BAC/JPM/C equity MAE by extending drivers or upgrading the forecasting architecture. _Status:_ ensemble weights now blend neural forecasts with proportional templates; equity MAE falls below $0.01$B across BAC/JPM/C (see `bank_ensemble.json`).
- [x] ML extension survey: catalogue candidate sequence/probabilistic approaches and prioritise experiments for the next milestone. _Status:_ roadmap recorded in `reports/q1/notes/ml_extension_roadmap.md`.
- [x] Simulation framing: document exogenous driver assumptions and sampling strategy for stochastic rollouts. _Status:_ simulation strategy captured in `reports/q1/notes/simulation_strategy.md`.

Part 1 is now complete; remaining work focuses on polishing documentation artifacts and bonus-question explorations.

### Part 2 — LLM-Assisted Analysis
- **LLM evaluation:** Select models (e.g., GPT-4o, Claude) for PDF-based financial analysis; record API versions and reproducibility notes.【F:Intern interview 2026 question 1.txt†L47-L61】 _Status:_ `mlcoe_q1.pipelines.run_llm_adapter` runs local `t5-small` baselines via HuggingFace with reproducible seeds, truncation-aware prompting, and responses captured in `reports/q1/artifacts/llm_responses_t5.parquet`.
- **Benchmarking vs. structured pipeline:** Compare LLM forecasts against the deterministic model using identical data slices.【F:Intern interview 2026 question 1.txt†L52-L55】 _Status:_ `mlcoe_q1.pipelines.compare_llm_and_forecaster` aligns the forecaster evaluation parquet with LLM response metrics, emits NaN-safe ticker summaries, and persists the merged tables for downstream analysis.
- **Ensembling:** Prototype combinations of LLM outputs and driver projections to assess complementary strengths.【F:Intern interview 2026 question 1.txt†L55-L56】 _Status:_ CFO guidance combines bank-ensemble MAE diagnostics with LLM coverage/error metrics via `mlcoe_q1.pipelines.generate_cfo_recommendations`, providing actionable blends of structured and unstructured signals (numerical ensembling extensions documented in the ML roadmap).
- **CFO recommendations:** Produce narrative guidance per company grounded in both quantitative outputs and LLM insights.【F:Intern interview 2026 question 1.txt†L56-L58】 _Status:_ Markdown reports in `reports/q1/status/cfo_recommendations.md` summarise prioritized tickers, deterministic MAE trends, and LLM diagnostics.
- **PDF automation:** Generalise the existing extractor beyond GM by accommodating LVMH/Tencent layouts and logging metadata for robustness checks.【F:Intern interview 2026 question 1.txt†L58-L65】 _Status:_ Built-in presets now cover GM/LVMH/Tencent with provenance metadata and unit-tested parsing heuristics for numeric extraction/label matching; additional issuers remain an optional enhancement.
- **Robustness:** Stress-test extraction stability across multiple runs or tools, documenting variability.【F:Intern interview 2026 question 1.txt†L61-L63】 _Status:_ Prompt truncation guards, NaN-safe summaries, and expanded pytest coverage (LLM adapter, CFO reporting, PDF parsing) address stability concerns; future robustness sweeps are tracked in the backlog.

### Bonus Tracks
- **Credit rating prototype:** Define the modelling approach, data sourcing, and Evergrande case study as described in Bonus Question 1.【F:Intern interview 2026 question 1.txt†L67-L83】
- **Risk warning extraction:** Design NLP pipelines to capture qualified opinions and other red-flag disclosures from annual reports (Bonus Question 2).【F:Intern interview 2026 question 1.txt†L84-L95】
- **Loan pricing model:** Survey literature, identify datasets, and scope extensions for illiquid borrowers per Bonus Question 3.【F:Intern interview 2026 question 1.txt†L97-L117】

## Current Coverage Snapshot

| Prompt Area | Status | Existing Assets |
| --- | --- | --- |
| Deterministic balance-sheet projection | ✅ Baseline implemented via `BalanceSheetState` and `project_forward` with accounting identity reconciliation; literature guidance distilled for modelling guardrails. | `mlcoe_q1/models/balance_sheet_constraints.py`, training/evaluation pipelines, `reports/q1/literature_summary.md` |
| Data ingestion & processing | ✅ CLI ingest + parquet conversion operational for GM/JPM/MSFT/AAPL. | `mlcoe_q1/pipelines/download_statements.py`, `mlcoe_q1/data/statement_processing.py` |
| Driver-based ML forecaster | ✅ MLP forecaster trains on log-scaled, per-asset, growth, and lagged drivers across nine tickers with auxiliary bank flags; roadmap for sequence/probabilistic upgrades documented. | `mlcoe_q1/pipelines/train_forecaster.py`, `mlcoe_q1/models/tf_forecaster.py`, `mlcoe_q1/utils/driver_features.py`, `reports/q1/notes/ml_extension_roadmap.md` |
| Bank-specific handling | ✅ Bank ensemble calibration blends neural predictions with proportional templates, driving BAC/JPM/C equity MAE below $0.01$B while preserving accounting identities. | `mlcoe_q1/models/bank_template.py`, `mlcoe_q1/models/bank_ensemble.py`, `mlcoe_q1/pipelines/calibrate_bank_ensemble.py`, `reports/q1/artifacts/forecaster_eval.parquet` |
| PDF ratio extraction | ⚠️ GM, LVMH, and Tencent configs available with provenance logging; broader issuer coverage still outstanding. | `mlcoe_q1/pipelines/extract_pdf_ratios.py` |
| LLM experimentation | ⚠️ Prompt dataset builder, baseline responder, HuggingFace adapter, evaluator, comparison, and CFO recommendation CLIs operational; expand coverage beyond the initial `t5-small` baseline and improve robustness metrics. | `mlcoe_q1/pipelines/build_llm_prompt_dataset.py`, `mlcoe_q1/pipelines/run_llm_adapter.py`, `mlcoe_q1/pipelines/evaluate_llm_responses.py`, `mlcoe_q1/pipelines/compare_llm_and_forecaster.py`, `mlcoe_q1/pipelines/generate_cfo_recommendations.py` |
| Bonus questions | ❌ Not started beyond literature notes. | — |

### Status Checkpoint — Incomplete Areas
- **Prompt coverage:** Part 1 and the Part 2 core deliverables are complete; ongoing work focuses on optional robustness sweeps, broader LLM portfolios, and bonus-track exploration.
- **Validation backlog:** Bank equity MAE now falls below $0.01$B via ensemble calibration; future work targets probabilistic extensions and longer-horizon backtests.
- **Reporting:** Final narrative packaging and broader issuer PDF coverage remain on the backlog alongside bonus questions.

## Near-Term Priorities (Next Milestone)
1. **Stabilise PDF tooling:** Validate extractor presets across additional issuers and add regression tests so ratio parsing is robust beyond GM/LVMH/Tencent.
2. **Probabilistic extensions:** Implement the roadmap items (GRU head, variational layers) to quantify forecast uncertainty and backtest multi-step scenarios.
3. **LLM coverage:** Broaden adapter support beyond `t5-small`, improve prompt truncation handling, and benchmark response robustness across multiple seeds.
4. **Reporting polish:** Consolidate the mathematical specification, simulation strategy, and CFO recommendation outputs into the final deliverable package.

## Research Backlog
- Read and annotate Pareja (2007/2009/2010) and Mejía-Pelaez & Vélez-Pareja (2011) for incorporation into the modelling note.
- Download GM/LVMH/Tencent annual reports and catalogue table structures for extractor configuration.
- Survey recent literature on LLMs for financial statement analysis (starting with Kim et al. 2024) to guide Part 2 architecture.
- Identify public datasets for credit rating modelling (e.g., S&P Capital IQ alternatives, Moody’s EDGAR releases) and assess licensing.

