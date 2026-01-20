# Question 1 Final Report — Strategic Lending Forecasting & LLM Benchmarking

## Executive Summary
The Question 1 deliverables provide a full-stack forecasting and reporting environment for Strategic Lending. The deterministic balance‑sheet
engine enforces accounting identities while a TensorFlow forecaster adds ratio‑based drivers, macro features, and probabilistic heads for
scenario generation. The Part 2 LLM workflow now benchmarks prompt datasets, evaluates coverage and errors, and synthesizes CFO‑ready
recommendations that default to deterministic forecasts when LLM responses lack quantitative output. Recent benchmarking runs introduce
structured JSON output enforcement and higher token budgets, lifting numeric coverage for `gpt-4o-mini` to over 80% across 99 prompts. This
report consolidates Part 1 and Part 2 findings, documents the chosen methods, and records testing evidence for correctness and reproducibility.

## 1. Literature Review Highlights
Key literature themes that shaped the solution:
- **Identity‑preserving projections.** Vélez‑Pareja & Mejía‑Pelaez informed the asset‑liability‑equity tie‑out and financing‑plug logic.
- **Driver‑based forecasting.** Financial‑ratio and growth‑driver approaches motivated the feature design (liquidity, leverage, profitability,
  spreads, and lagged deltas) for forward projections.
- **Probabilistic scenario modelling.** Bayesian/variational forecasting and ensemble techniques informed the Gaussian and variational heads
  plus calibrated bank ensembles for uncertainty.
- **LLM quantitative benchmarking.** Recent evaluation frameworks for structured numeric outputs informed coverage/MAE/MAPE evaluation.

The repository includes a dedicated literature summary and specification documents to preserve citation notes and design rationale.

## 2. Method Selection & Rationale
The project combines deterministic constraints with learned forecasting:
- **Deterministic balance‑sheet engine** ensures Assets = Liabilities + Equity and creates a reliable backbone for underwriting use.
- **Neural forecaster** adds flexibility to model nonlinear relationships between ratios, growth, and macro history, while probabilistic heads
  enable scenario ranges.
- **Calibrated bank ensembles** incorporate bank‑specific liability mix priors to stabilize outputs for financial institutions.
- **LLM benchmarking** provides a transparent assessment of model coverage and error before introducing LLMs into a regulated forecasting
  workflow.

This hybrid strategy balances interpretability (deterministic accounting) with predictive power (ML/LLM augmentations).

## 3. Data Acquisition & Feature Engineering
- **Data sources.** Financial statements for nine tickers (AAPL, MSFT, GM, JPM, BAC, C, HON, CAT, UNP) are downloaded via reproducible
  pipelines and cached as raw JSON.
- **Feature construction.** The driver dataset blends ratio metrics (liquidity, leverage, profitability), growth signals, and lagged covariates.
- **Validation.** Dataset validation checks filing gaps, duplicates, and missing columns prior to training and evaluation.

## 4. Part 1 Results — Forecasting & Identity Preservation
### 4.1 Deterministic + ML Forecasting
The deterministic constraint layer reconciles financing gaps and preserves accounting identities across forecast horizons. The TensorFlow
forecaster produces multi‑target balance‑sheet predictions with bank‑aware routing and shared feature towers.

### 4.2 Forecast Accuracy (Latest Two Statement Pairs)
Mean absolute errors (billions USD) and identity gaps averaged over the latest two statement pairs:

| Ticker | Mode | Assets MAE (B) | Equity MAE (B) | Identity Gap (B) |
| --- | --- | ---: | ---: | ---: |
| AAPL | mlp | 3.58 | 17.90 | 0.00 |
| BAC | bank_template | 106.40 | 383.35 | 0.00 |
| C | bank_template | 19.19 | 355.92 | 0.00 |
| CAT | mlp | 11.33 | 7.26 | 0.00 |
| GM | mlp | 41.84 | 7.03 | 0.00 |
| HON | mlp | 19.44 | 3.12 | N/A |
| JPM | bank_template | 144.24 | 606.87 | 0.00 |
| MSFT | mlp | 27.83 | 64.07 | 0.00 |
| UNP | mlp | 29.33 | 0.82 | 9.88 |

_Notes: identity gaps are near zero for most issuers, but UNP shows a 9.88B gap; HON’s identity gap is undefined due to missing liabilities
splits in the source statement. Net‑income MAE is not reported in the latest evaluation artifact._

### 4.3 PDF Ratio Extraction
PDF ratio extraction workflows now support multiple issuer layouts (GM, LVMH, Tencent, Alibaba, JPM, Exxon, Microsoft, VW, Google),
including text‑fallback parsing when table extraction fails. Provenance metadata logs source PDFs, page indexes, and table strategies to
support downstream CFO narratives.

## 5. Part 2 Results — LLM Benchmarking & Reporting
- **Prompt datasets** pair structured statements with forecast targets.
- **LLM evaluation** computes numeric coverage, MAE, and MAPE; baseline `t5-small` produces zero numeric coverage for AAPL/BAC prompts,
  reinforcing the need for stronger models before reliance.
- **Comparison tooling** aligns deterministic forecasts with LLM outputs and generates recommendation memos for executive review.

Latest LLM benchmark summary (99 prompts, 3 seeds):

| Model | Max New Tokens | Coverage Mean | MAE Mean (B) | MAPE Mean |
| --- | ---: | ---: | ---: | ---: |
| gpt-4o-mini (openai-chat) | 2048 | 81.27% | 9.95 | 1.72 |

## 6. Testing Plan & Results
Testing follows a layered strategy:
- **Unit tests** for driver features, bank template logic, and constraint enforcement.
- **Integration tests** for ingestion, training, evaluation, PDF extraction, and LLM pipelines.
- **Reproducibility checks** for seeded datasets and deterministic reporting outputs.

Current test coverage includes:
- `pytest tests/mlcoe_q1 -q` (unit + integration for data pipelines, models, and reporting).

## 7. Conclusion & Next Steps
The Question 1 stack now meets the Part 1 forecasting requirements and establishes a defensible LLM benchmarking framework for Part 2.
Recommended follow‑ups:
- Expand LLM coverage with stronger models and prompt variants.
- Increase PDF ratio presets for additional issuers.
- Extend probabilistic scenario support with macro‑driven Monte Carlo overlays.

## 8. Bonus Questions — Credit Rating, Risk Warnings, Loan Pricing
### 8.1 Part B (Bonus Q1) — Credit Rating Model + Shenanigans Checks
- **Model form.** We implement an Altman Z‑score model (`mlcoe_q1/credit/altman.py`) that maps balance‑sheet and income‑statement ratios
  into ordinal rating buckets (investment‑grade → ccc).
- **Training data.** Features are built from Yahoo Finance statements for the portfolio tickers plus Evergrande (`3333.HK`), stored in
  `mlcoe_q1/data/credit_ratings/altman_features.parquet` with metadata in `mlcoe_q1/data/credit_ratings/altman_features.json`.
- **Evergrande case study.** The 2022 filing yields a Z‑score of −0.92 (rating bucket `ccc`) with negative working capital and retained
  earnings, documented in `reports/q1/artifacts/evergrande_credit_rating.json`.
- **Shenanigans validation.** We test bankrupt‑company annual reports (BBBY and Sears) using the risk‑warning scanner described below;
  both filings trigger **going concern** and other disclosure flags, demonstrating the tool’s ability to surface warning signals.

### 8.2 Part C (Bonus Q2) — Risk Warning Extraction
- **Engine.** A keyword‑driven extractor scans annual‑report text chunks for high‑risk disclosures (going concern, liquidity, regulatory,
  covenant breaches, etc.) and emits structured warnings with snippets.
- **Bankrupt‑company test.** SEC 10‑K filings for BBBY (2023) and Sears (2017) were ingested from HTML and chunked into 524 passages; the
  extractor identified 32 warnings (20 going‑concern hits for BBBY; 2 for Sears), recorded in
  `reports/q1/artifacts/risk_warnings.parquet` and summarised in `reports/q1/artifacts/risk_warnings_summary.json`.

### 8.3 Part D (Bonus Q3) — Loan Pricing Model
- **Approach.** A term‑loan pricing engine computes spreads over Treasury yields using Altman‑derived credit buckets, leverage, and macro
  overlays, producing baseline/optimistic/stress scenarios.
- **Results.** Average all‑in rates are 8.73% (baseline), 7.13% (optimistic), and 10.33% (stress), with full tables in
  `reports/q1/artifacts/loan_pricing.parquet` and summary stats in `reports/q1/artifacts/loan_pricing_summary.json`.

## Appendix — Key References & Artifacts
- Deterministic specification: `reports/q1/deterministic_balance_sheet_spec.md` 
- Literature summary: `reports/q1/literature_summary.md` 
- Interim dashboards: `reports/q1/status/` 
- LLM comparison artifacts: `reports/q1/q1_response_summary.md`
- Credit rating dataset: `mlcoe_q1/data/credit_ratings/altman_features.parquet`
- Evergrande case study: `reports/q1/artifacts/evergrande_credit_rating.json`
- Risk warning summary: `reports/q1/artifacts/risk_warnings_summary.json`
- Loan pricing summary: `reports/q1/artifacts/loan_pricing_summary.json`
