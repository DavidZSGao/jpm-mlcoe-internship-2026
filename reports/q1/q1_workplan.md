# Question 1 Execution Plan

This document maps the internship prompt requirements to the current repository assets and highlights the next set of concrete deliverables.

## Prompt Decomposition

### Part 1 — Balance Sheet Modelling
- **Literature grounding:** Summarise the accounting identity approach from Vélez-Pareja (2007, 2009, 2010) and Mejía-Pelaez & Vélez-Pareja (2011); extract the minimal equation set required for deterministic balance-sheet projection respecting `Assets = Liabilities + Equity` and related relationships.【F:Intern interview 2026 question 1.txt†L19-L36】 _Status:_ captured in `reports/q1/literature_summary.md` with implementation takeaways.
- **Mathematical specification:** Formalise the evolution equations for balance-sheet fields with explicit handling of dependent accounts (e.g., working-capital tie-outs, debt schedules). Provide a note on framing the problem as a time series and the identity management strategy.【F:Intern interview 2026 question 1.txt†L26-L33】
- **TensorFlow implementation:** Translate the specification into trainable modules layered on top of the existing `BalanceSheetState` / `project_forward` machinery; ensure gradients propagate through constraint adjustments.【F:Intern interview 2026 question 1.txt†L31-L33】
- **Data acquisition:** Extend the `yfinance` ingestion CLI to cover the tickers required for experiments; cache raw JSON responses for reproducibility.【F:Intern interview 2026 question 1.txt†L33-L35】
- **Training and evaluation:** Define experiment splits, loss metrics, and backtests to validate the forecasts and accounting identity compliance.【F:Intern interview 2026 question 1.txt†L35-L42】
- **Earnings linkage:** Document whether the driver/constraint framework can forecast earnings and what additional modelling is necessary.【F:Intern interview 2026 question 1.txt†L42-L43】
- **ML extensions:** Catalogue candidate techniques (seq2seq, normalising flows, probabilistic forecasting) suitable for improving the baseline.【F:Intern interview 2026 question 1.txt†L43-L44】
- **Simulation view:** Identify the exogenous variables `x(t)` to support stochastic simulations consistent with the hint in the prompt.【F:Intern interview 2026 question 1.txt†L44-L46】

### Part 2 — LLM-Assisted Analysis
- **LLM evaluation:** Select models (e.g., GPT-4o, Claude) for PDF-based financial analysis; record API versions and reproducibility notes.【F:Intern interview 2026 question 1.txt†L47-L61】
- **Benchmarking vs. structured pipeline:** Compare LLM forecasts against the deterministic model using identical data slices.【F:Intern interview 2026 question 1.txt†L52-L55】
- **Ensembling:** Prototype combinations of LLM outputs and driver projections to assess complementary strengths.【F:Intern interview 2026 question 1.txt†L55-L56】
- **CFO recommendations:** Produce narrative guidance per company grounded in both quantitative outputs and LLM insights.【F:Intern interview 2026 question 1.txt†L56-L58】
- **PDF automation:** Generalise the existing extractor beyond GM by accommodating LVMH/Tencent layouts and logging metadata for robustness checks.【F:Intern interview 2026 question 1.txt†L58-L65】
- **Robustness:** Stress-test extraction stability across multiple runs or tools, documenting variability.【F:Intern interview 2026 question 1.txt†L61-L63】

### Bonus Tracks
- **Credit rating prototype:** Define the modelling approach, data sourcing, and Evergrande case study as described in Bonus Question 1.【F:Intern interview 2026 question 1.txt†L67-L83】
- **Risk warning extraction:** Design NLP pipelines to capture qualified opinions and other red-flag disclosures from annual reports (Bonus Question 2).【F:Intern interview 2026 question 1.txt†L84-L95】
- **Loan pricing model:** Survey literature, identify datasets, and scope extensions for illiquid borrowers per Bonus Question 3.【F:Intern interview 2026 question 1.txt†L97-L117】

## Current Coverage Snapshot

| Prompt Area | Status | Existing Assets |
| --- | --- | --- |
| Deterministic balance-sheet projection | ✅ Baseline implemented via `BalanceSheetState` and `project_forward` with accounting identity reconciliation; literature guidance distilled for modelling guardrails. | `mlcoe_q1/models/balance_sheet_constraints.py`, training/evaluation pipelines, `reports/q1/literature_summary.md` |
| Data ingestion & processing | ✅ CLI ingest + parquet conversion operational for GM/JPM/MSFT/AAPL. | `mlcoe_q1/pipelines/download_statements.py`, `mlcoe_q1/data/statement_processing.py` |
| Driver-based ML forecaster | ⚠️ Prototype MLP now ingests log-revenue and asset-scaled drivers across nine tickers; further sector conditioning and richer temporal context still pending. | `mlcoe_q1/pipelines/train_forecaster.py`, `mlcoe_q1/models/tf_forecaster.py`, `mlcoe_q1/utils/driver_features.py` |
| Bank-specific handling | ⚠️ Proportional template calibrated with label-normalised inputs and liability ratios tied to the latest filings; JPM equity MAE now ~6×10^11 (down from ~8×10^11) yet still needs richer banking drivers. | `mlcoe_q1/models/bank_template.py`, `mlcoe_q1/models/artifacts/driver_forecaster/bank_templates.json` |
| PDF ratio extraction | ⚠️ GM, LVMH, and Tencent configs available with provenance logging; broader issuer coverage still outstanding. | `mlcoe_q1/pipelines/extract_pdf_ratios.py` |
| LLM experimentation | ❌ Not started; no API integration or benchmarking yet. | — |
| Bonus questions | ❌ Not started beyond literature notes. | — |

## Near-Term Priorities (Next Milestone)
1. **Stabilise PDF tooling:** Add configuration abstraction and table-parsing heuristics to support LVMH; capture provenance (model/tool version, extraction timestamp) alongside ratios. _Status:_ built-in CLI now emits metadata, accepts JSON overrides, and ships GM/LVMH/Tencent presets; next step is validating additional issuers via automated tests.
2. **Scale driver dataset:** ✅ Normalised features (log revenues, per-asset scaling) now flow through the driver dataset with HON/CAT/UNP/BAC/C additions; next step is to backfill remaining target tickers and validate temporal alignment before training deeper models.
3. **Forecaster calibration:** Add sector-specific heads or richer banking covariates to push JPM/BAC/C errors below current ~6×10^11 equity MAE despite the improved templates. A new `summarize_forecaster_eval` CLI now aggregates MAE identity-gap statistics to monitor progress ticker-by-ticker.
4. **Documentation uplift:** Draft the mathematical specification section summarising the constraint system and simulation framing for Part 1 write-up. _Status:_ Initial specification captured in `reports/q1/deterministic_balance_sheet_spec.md`; cross-referenced literature notes in `reports/q1/literature_summary.md`.

## Research Backlog
- Read and annotate Pareja (2007/2009/2010) and Mejía-Pelaez & Vélez-Pareja (2011) for incorporation into the modelling note.
- Download GM/LVMH/Tencent annual reports and catalogue table structures for extractor configuration.
- Survey recent literature on LLMs for financial statement analysis (starting with Kim et al. 2024) to guide Part 2 architecture.
- Identify public datasets for credit rating modelling (e.g., S&P Capital IQ alternatives, Moody’s EDGAR releases) and assess licensing.

