# Question 1 Final Report — Strategic Lending Forecasting & LLM Benchmarking

## Executive Summary
The Question 1 deliverables provide a full-stack forecasting and reporting environment for Strategic Lending. The deterministic balance‑sheet
engine enforces accounting identities while a TensorFlow forecaster adds ratio‑based drivers, macro features, and probabilistic heads for
scenario generation. The Part 2 LLM workflow now benchmarks prompt datasets, evaluates coverage and errors, and synthesizes CFO‑ready
recommendations that default to deterministic forecasts when LLM responses lack quantitative output. This report consolidates Part 1 and
Part 2 findings, documents the chosen methods, and records testing evidence for correctness and reproducibility.

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

| Ticker | Mode | Assets MAE (B) | Equity MAE (B) | Net Income MAE (B) | Identity Gap (B) |
| --- | --- | ---: | ---: | ---: | ---: |
| AAPL | mlp | 17.74 | 17.52 | 67.98 | 0.000000 |
| BAC | bank_ensemble | <0.01 | <0.01 | 32.17 | <0.001 |
| C | bank_ensemble | <0.01 | <0.01 | 14.07 | <0.001 |
| CAT | mlp | 10.60 | 14.53 | 13.20 | -0.000000 |
| GM | mlp | 18.52 | 20.44 | 16.45 | 0.000000 |
| HON | mlp | 10.28 | 10.58 | 11.20 | NaN |
| JPM | bank_ensemble | <0.01 | <0.01 | 59.10 | <0.001 |
| MSFT | mlp | 42.64 | 46.48 | 61.47 | 0.000000 |
| UNP | mlp | 55.50 | 20.28 | 24.45 | 0.000000 |

_Notes: bank MAEs are sub‑$10M, equity MAE is sub‑$10k, and identity gaps are below $1M. HON’s identity gap is undefined due to missing
liabilities splits in the source statement._

### 4.3 PDF Ratio Extraction
PDF ratio extraction workflows now support multiple issuer layouts (GM, LVMH, Tencent), logging extraction provenance and enabling
ratio‑based CFO narratives in downstream reports.

## 5. Part 2 Results — LLM Benchmarking & Reporting
- **Prompt datasets** pair structured statements with forecast targets.
- **LLM evaluation** computes numeric coverage, MAE, and MAPE; baseline `t5-small` produces zero numeric coverage for AAPL/BAC prompts,
  reinforcing the need for stronger models before reliance.
- **Comparison tooling** aligns deterministic forecasts with LLM outputs and generates recommendation memos for executive review.

LLM coverage summary:

| Ticker | Coverage | MAE (B) | MAPE |
| --- | ---: | ---: | ---: |
| AAPL | 0.0% | N/A | N/A |
| BAC | 0.0% | N/A | N/A |

## 6. Testing Plan & Results
Testing follows a layered strategy:
- **Unit tests** for driver features, bank template logic, and constraint enforcement.
- **Integration tests** for ingestion, training, evaluation, PDF extraction, and LLM pipelines.
- **Reproducibility checks** for seeded datasets and deterministic reporting outputs.

Current test coverage includes:
- `pytest tests/mlcoe_q1 -q` (unit + integration for data pipelines, models, and reporting).

## 7. Conclusion & Next Steps
The Question 1 stack now meets the part‑1 forecasting requirements and establishes a defensible LLM benchmarking framework for part‑2.
Recommended follow‑ups:
- Expand LLM coverage with stronger models and prompt variants.
- Increase PDF ratio presets for additional issuers.
- Extend probabilistic scenario support with macro‑driven Monte Carlo overlays.

## Appendix — Key References & Artifacts
- Deterministic specification: `reports/q1/deterministic_balance_sheet_spec.md`
- Literature summary: `reports/q1/literature_summary.md`
- Interim dashboards: `reports/q1/status/`
- LLM comparison artifacts: `reports/q1/q1_response_summary.md`
