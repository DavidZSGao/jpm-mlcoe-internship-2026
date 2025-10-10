# Question 1 — Interim Report

_Prepared ahead of the November Part 1 submission checkpoint to capture the current state of deterministic and LLM deliverables for Question 1._

## 1. Executive Summary
- Delivered a deterministic balance-sheet projection stack that enforces accounting identities, augments ratio-based drivers with growth and lagged covariates, and trains a TensorFlow forecaster backed by calibrated bank ensembles for BAC, JPM, and C.【F:reports/q1/deterministic_balance_sheet_spec.md†L1-L91】【F:mlcoe_q1/models/tf_forecaster.py†L1-L310】【F:mlcoe_q1/models/bank_ensemble.py†L1-L193】
- Processed nine ticker histories via reproducible Yahoo Finance ingestion, validation, and feature pipelines, then evaluated forecasts across assets, equity, earnings, and identity gaps with summarisation/reporting CLIs and pytest coverage.【F:mlcoe_q1/data/yfinance_ingest.py†L1-L98】【F:mlcoe_q1/pipelines/build_driver_dataset.py†L1-L268】【F:mlcoe_q1/pipelines/evaluate_forecaster.py†L1-L303】【F:tests/mlcoe_q1/test_validate_driver_dataset.py†L1-L113】
- Built Part 2 LLM tooling (prompt dataset, HuggingFace adapter, evaluation/comparison CLIs, CFO recommendations) and stress-tested PDF ratio extraction for GM, LVMH, and Tencent, highlighting that baseline `t5-small` responses lack quantitative coverage and should be combined with deterministic outputs.【F:mlcoe_q1/pipelines/build_llm_prompt_dataset.py†L1-L219】【F:mlcoe_q1/pipelines/run_llm_adapter.py†L1-L108】【F:mlcoe_q1/pipelines/evaluate_llm_responses.py†L1-L295】【F:mlcoe_q1/pipelines/compare_llm_and_forecaster.py†L1-L329】【F:mlcoe_q1/pipelines/generate_cfo_recommendations.py†L1-L164】【F:mlcoe_q1/pipelines/extract_pdf_ratios.py†L1-L226】

## 2. Literature & Problem Framing
- Vélez-Pareja and Mejía-Pelaez papers guided the identity-preserving projection design; key takeaways and implementation notes are summarised in `reports/q1/literature_summary.md` for quick reference.【F:reports/q1/literature_summary.md†L1-L88】
- The deterministic balance-sheet specification formalises asset/liability/equity evolution, financing plugs, and simulation framing, enabling direct translation into TensorFlow layers.【F:reports/q1/deterministic_balance_sheet_spec.md†L1-L91】
- Simulation and ML extension roadmaps capture exogenous driver selection, probabilistic upgrades, and future research backlog for continued development.【F:reports/q1/notes/simulation_strategy.md†L1-L51】【F:reports/q1/notes/ml_extension_roadmap.md†L1-L53】

## 3. Data Acquisition & Processing
- Raw statements for AAPL, MSFT, GM, JPM, BAC, C, HON, CAT, and UNP are cached under `mlcoe_q1/data/raw/*.json` using the `download_statements` CLI to guarantee reproducibility.【F:mlcoe_q1/data/yfinance_ingest.py†L1-L98】
- Processed balance-sheet/income features are produced via `prepare_processed_data`, while `build_driver_dataset` constructs configurable ratio, growth, and lagged covariates with optional lag filling controls.【F:mlcoe_q1/pipelines/build_driver_dataset.py†L1-L268】
- The `validate_driver_dataset` CLI checks for duplicate periods, missing columns, sparse histories, and filing gaps; its pytest suite exercises both success and failure paths to keep data hygiene auditable.【F:mlcoe_q1/pipelines/validate_driver_dataset.py†L1-L207】【F:tests/mlcoe_q1/test_validate_driver_dataset.py†L1-L113】

## 4. Modelling Approach
### 4.1 Deterministic Projection Layer
- `balance_sheet_constraints.project_forward` enforces Assets = Liabilities + Equity and related tie-outs, reconciling financing gaps after each forecast step.【F:mlcoe_q1/models/balance_sheet_constraints.py†L1-L222】

### 4.2 Feature Engineering & Forecaster Architecture
- Driver features include leverage, liquidity, profitability, tangible equity, interest spreads, and year-over-year growth, with lag augmentation handled by `augment_with_lagged_features` for temporal context.【F:mlcoe_q1/utils/driver_features.py†L1-L270】
- The TensorFlow forecaster combines shared MLP towers with auxiliary sector signals, bank-indicator routing, and complement heads so bank and industrial predictions coexist within a single serialized model.【F:mlcoe_q1/models/tf_forecaster.py†L1-L310】
- Training (`train_forecaster.py`) and evaluation (`evaluate_forecaster.py`) pipelines persist scaling metadata, register custom layers, and emit rich parquet diagnostics for downstream tooling.【F:mlcoe_q1/pipelines/train_forecaster.py†L1-L293】【F:mlcoe_q1/pipelines/evaluate_forecaster.py†L1-L303】

### 4.3 Bank Ensemble Calibration
- Proportional templates (`bank_template.py`) capture liability mix priors per bank, while `BankForecastEnsemble` blends template outputs with neural predictions using calibrated weights saved in `bank_ensemble.json` and auto-applied during evaluation.【F:mlcoe_q1/models/bank_template.py†L1-L177】【F:mlcoe_q1/models/bank_ensemble.py†L1-L193】【F:mlcoe_q1/pipelines/calibrate_bank_ensemble.py†L1-L210】

## 5. Forecast Evaluation
Mean absolute errors (billions USD) and identity gaps (billions) averaged over the latest two statement pairs are summarised below; banks use the calibrated ensemble mode.

| Ticker | Mode | Assets MAE (B) | Equity MAE (B) | Net Income MAE (B) | Identity Gap (B) |
| --- | --- | ---: | ---: | ---: | ---: |
| AAPL | mlp | 17.74 | 17.52 | 67.98 | 0.000000 |
| BAC | bank_ensemble | 0.00 | 0.00 | 32.17 | -0.000000 |
| C | bank_ensemble | 0.00 | 0.00 | 14.07 | 0.000000 |
| CAT | mlp | 10.60 | 14.53 | 13.20 | -0.000000 |
| GM | mlp | 18.52 | 20.44 | 16.45 | 0.000000 |
| HON | mlp | 10.28 | 10.58 | 11.20 | NaN |
| JPM | bank_ensemble | 0.00 | 0.00 | 59.10 | 0.000000 |
| MSFT | mlp | 42.64 | 46.48 | 61.47 | 0.000000 |
| UNP | mlp | 55.50 | 20.28 | 24.45 | 0.000000 |

_Bank equity MAE falls below $10k (displayed as 0.00 B), while identity gaps remain numerically zero across evaluated tickers; HON’s identity gap is undefined because the source statement omits the necessary liabilities split._【F:reports/q1/status/forecaster_status.md†L1-L24】【F:reports/q1/status/cfo_recommendations.md†L1-L24】【F:reports/q1/q1_interim_report.md†L26-L38】

## 6. PDF Ratio Extraction
- `extract_pdf_ratios` supports GM, LVMH, and Tencent layouts with JSON-configurable strategies, pdfplumber-backed parsing, provenance logging, and pytest-backed heuristics for stable numeric extraction.【F:mlcoe_q1/pipelines/extract_pdf_ratios.py†L1-L226】【F:tests/mlcoe_q1/test_extract_pdf_ratios.py†L1-L24】
- Ratio outputs feed the comparison CLI and documentation for CFO-facing narratives, demonstrating automated computation of net income, cost-to-income, liquidity, leverage, and coverage metrics directly from filings.【F:mlcoe_q1/pipelines/compare_ratio_sources.py†L1-L162】【F:reports/q1/literature_summary.md†L1-L88】

## 7. LLM Benchmarking & Recommendations
- Prompt datasets pair processed statements with ground-truth targets; the HuggingFace adapter runs truncation-aware inference (default `t5-small`) with seeded decoding, while `evaluate_llm_responses` computes coverage, MAE, and MAPE metrics per record.【F:mlcoe_q1/pipelines/build_llm_prompt_dataset.py†L1-L219】【F:mlcoe_q1/pipelines/run_llm_adapter.py†L1-L108】【F:mlcoe_q1/pipelines/evaluate_llm_responses.py†L1-L295】
- Baseline `t5-small` responses produced zero numeric coverage across AAPL and BAC, underscoring the need for stronger models before quantitative reliance. Coverage and error summary:

  | Ticker | Coverage | MAE (B) | MAPE |
  | --- | ---: | ---: | ---: |
  | AAPL | 0.0% | N/A | N/A |
  | BAC | 0.0% | N/A | N/A |

  _Coverage reflects the share of prompts where the model emitted parseable numeric forecasts; MAE/MAPE remain undefined when coverage is zero._【F:reports/q1/status/llm_vs_forecaster_t5.csv†L1-L4】【F:reports/q1/q1_interim_report.md†L42-L49】
- `compare_llm_and_forecaster` aligns structured and LLM metrics, and `generate_cfo_recommendations` produces Markdown narratives prioritising deterministic forecasts when LLM coverage is low.【F:mlcoe_q1/pipelines/compare_llm_and_forecaster.py†L1-L329】【F:mlcoe_q1/pipelines/generate_cfo_recommendations.py†L1-L164】【F:reports/q1/status/cfo_recommendations.md†L1-L35】

## 8. Testing & Reproducibility
- End-to-end pytest coverage spans driver features, TensorFlow layers, bank ensembles, PDF extraction, LLM adapters, evaluation pipelines, and reporting utilities (`pytest tests/mlcoe_q1 -q`).【F:tests/mlcoe_q1/test_tf_forecaster.py†L1-L64】【F:tests/mlcoe_q1/test_bank_ensemble.py†L1-L96】【F:tests/mlcoe_q1/test_run_llm_adapter.py†L1-L102】
- CLIs expose `--help` documentation and produce deterministic parquet/JSON artifacts, enabling reproducible reruns documented in the root README usage examples.【F:README.md†L31-L122】

## 9. Outstanding Opportunities
- Broaden PDF presets to additional issuers and add regression tests for multi-layout coverage.【F:reports/q1/q1_workplan.md†L92-L111】
- Upgrade LLM experiments with higher-coverage models, prompt variants, and quantitative ensembles, tracking robustness across seeds and versions.【F:reports/q1/q1_workplan.md†L112-L120】
- Pursue bonus tracks (credit rating, risk warnings, loan pricing) leveraging existing deterministic and LLM infrastructure as scaffolding.【F:reports/q1/q1_workplan.md†L122-L142】

## 10. Submission Checklist
- Codebase implemented in Python 3/TensorFlow with TensorFlow Probability ready for probabilistic extensions.【F:requirements.txt†L1-L35】
- Comprehensive testing (unit + integration) and documentation assets (`README`, execution plan, literature summaries, status dashboards) committed alongside reproducible data artifacts.【F:reports/q1/q1_workplan.md†L1-L142】【F:reports/q1/status/forecaster_status.md†L1-L24】【F:reports/q1/status/cfo_recommendations.md†L1-L35】
