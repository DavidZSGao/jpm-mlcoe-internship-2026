# Question 1 Roadmap — Strategic Lending Application

## High-Level Objectives
- Build a balance-sheet forecasting pipeline that preserves core accounting identities for a bank’s strategic lending desk.【F:Intern interview 2026 question 1 ver 2.txt†L5-L19】
- Implement TensorFlow-based simulators/surrogate models grounded in Velez-Pareja style formulations.
- Benchmark forecasts against baseline econometric/time-series approaches and document accuracy vs. constraint adherence.
- Prototype an LLM-assisted workflow for financial statement analysis and ratio extraction informed by Alonso (2024), Farr (2025), and Zhang (2025).【F:Intern interview 2026 question 1 ver 2.txt†L21-L34】
- Deliver actionable insights for a selected company, including qualitative recommendations.

## Workstreams & Immediate Tasks

### 1. Data Acquisition & Normalisation
- [x] Draft reusable Yahoo Finance ingestion utilities (`mlcoe_q1/data/yfinance_ingest.py`) to pull income statement & balance sheet time series.
- [x] Define canonical schema + storage format (parquet) within `data/processed/` with metadata hooks for accounting identities.
- [x] Seed dataset list (GM, JPM, MSFT, AAPL) and capture data availability diagnostics.
- [x] Wire CLI downloader entry point for batch ingestion (`mlcoe_q1/pipelines/download_statements.py`).
- [x] Wire CLI converter (`mlcoe_q1/pipelines/prepare_processed_data.py`).

### 2. Identity-Preserving Balance Sheet Model
- [x] Formalise equations following Vélez-Pareja (2009/2010) in a Python module (`mlcoe_q1/models/balance_sheet_constraints.py`).
- [x] Prototype a deterministic simulator mapping income statement drivers to balance sheet fields; confirm assets = liabilities + equity numerically.
- [x] Explore state-space / RNN formulation for temporal evolution with constraint enforcement (e.g., Lagrange multipliers, projection layer). _Status:_ implemented as the TensorFlow balance-sheet forecaster with projection-based reconciliation.

### 3. TensorFlow Training Pipeline
- [x] Set up TensorFlow model scaffolding (`mlcoe_q1/models/tf_forecaster.py`) enabling multi-step prediction and teacher forcing.
- [x] Implement baseline constraint backtest (`mlcoe_q1/pipelines/backtest_baseline.py`).
- [x] Implement training pipeline (`mlcoe_q1/pipelines/train_forecaster.py`) with configurable horizons, loss composition (MSE + identity penalty), and experiment logging.
- [x] Define evaluation metrics (RMSE per account, aggregate identity violation) in `mlcoe_q1/evaluation/metrics.py` and persist results to parquet for downstream reporting.
- [x] Next step: extend the TensorFlow forecaster with probabilistic heads (e.g., variational layers or quantile losses) per the Part 1 extension roadmap. _Status:_ Gaussian mean/log-variance head shipped with sampling support; upcoming work targets multi-step rollouts and alternative stochastic architectures (e.g., GRU-based encoders).

### 4. Forecast Validation & Reporting
- [x] Build experiment runners under `mlcoe_q1/experiments/` that output JSON summaries into `reports/q1/artifacts/` and plots into `reports/q1/figures/`.
- [x] Draft initial narrative template in `reports/q1/notes/` to capture findings & CFO recommendations (`reports/q1/status/cfo_recommendations.md`).
- [x] Integrate automated ratio computation utilities (quick ratio, D/E, etc.) for comparability against official filings.
- [x] Add `summarize_forecaster_evaluation.py` to roll the Monte Carlo-enabled evaluation parquet into lender-facing MAE/coverage tables by ticker/mode.
- [ ] Next step: consolidate the mathematical specification, simulation note, and CFO recommendations into a publishable briefing for the Strategic Lending Division.

### 5. LLM-Assisted Analysis (Part 2)
- [x] Select baseline LLM/API configuration (HuggingFace `t5-small`) and outline prompt templates for extracting financial insights; capture API/version metadata in pipeline configs.
- [x] Evaluate ensemble combinations: constrained TensorFlow model + LLM outputs via the CFO recommendation generator that merges MAE diagnostics with qualitative coverage signals.
- [x] Implement PDF parsing pipelines (GM, LVMH, Tencent, Alibaba, JPM, Exxon annual reports) leveraging `pdfplumber` with structured provenance logging.
- [ ] Next step: expand to higher-capacity hosted models (e.g., GPT-4o, Claude) and run robustness sweeps across multiple seeds and prompt truncation limits. _In-flight:_ HuggingFace causal adapters added (`--adapter hf-causal`) so decoder-only checkpoints can participate in the benchmark; hosted API benchmarking remains open.

### 6. Bonus Tracks (Backlog)
- [x] Credit rating classifier design + data sourcing (e.g., WRDS/SEC filings, rating agency datasets). Altman dataset builder and scoring CLI now cover the core lending portfolio plus Evergrande with JSON artifacts for case studies.
- [x] Risk warning extraction engine leveraging NLP/LLM summarisation; compile list of bankruptcy reports for testing. Delivered via `mlcoe_q1.pipelines.extract_risk_warnings` with keyword heuristics and Strategic Lending summaries; future work covers LLM-assisted red-flag classification.
- [x] Loan pricing model survey, dataset curation, and uncertainty quantification workflow. Loan pricing utilities translate scenario quantiles and Altman ratings into decomposed spreads, and the `price_loans` pipeline now supports macro-sensitivity configs to layer policy-rate or unemployment shocks into borrower rates alongside configurable risk-free/spread tables.

## Cross-Cutting Considerations
- Maintain modular structure mirroring `mlcoe_q2` for consistency across questions.
- Ensure all scripts accept CLI arguments for tickers/date ranges to ease automation.
- Version control datasets (checksums/metadata) for replicability; store large raw files outside repo if needed and document download scripts.
- Align reporting artifacts with existing conventions (`reports/q2/…`) for unified final submission.
