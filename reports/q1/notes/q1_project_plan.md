# Question 1 Roadmap — Strategic Lending Application

## High-Level Objectives
- Build a balance-sheet forecasting pipeline that preserves core accounting identities.
- Implement TensorFlow-based simulators/surrogate models grounded in Velez-Pareja style formulations.
- Benchmark forecasts against baseline econometric/time-series approaches and document accuracy vs. constraint adherence.
- Prototype an LLM-assisted workflow for financial statement analysis and ratio extraction (Kim 2024).
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
- [ ] Explore state-space / RNN formulation for temporal evolution with constraint enforcement (e.g., Lagrange multipliers, projection layer).

### 3. TensorFlow Training Pipeline
- [ ] Set up TensorFlow model scaffolding (`mlcoe_q1/models/tf_forecaster.py`) enabling multi-step prediction and teacher forcing.
- [x] Implement baseline constraint backtest (`mlcoe_q1/pipelines/backtest_baseline.py`).
- [ ] Implement training pipeline (`mlcoe_q1/pipelines/train_forecaster.py`) with configurable horizons, loss composition (MSE + identity penalty), and experiment logging.
- [ ] Define evaluation metrics (RMSE per account, aggregate identity violation) in `mlcoe_q1/evaluation/metrics.py`.

### 4. Forecast Validation & Reporting
- [ ] Build experiment runners under `mlcoe_q1/experiments/` that output JSON summaries into `reports/q1/artifacts/` and plots into `reports/q1/figures/`.
- [ ] Draft initial narrative template in `reports/q1/notes/` to capture findings & CFO recommendations.
- [ ] Integrate automated ratio computation utilities (quick ratio, D/E, etc.) for comparability against official filings.

### 5. LLM-Assisted Analysis (Part 2)
- [ ] Select target LLM/API (e.g., `gpt-4o-2024-08-06`) and outline prompt templates for extracting financial insights.
- [ ] Evaluate ensemble combinations: constrained TensorFlow model + LLM outputs (e.g., weighted averaging or residual correction).
- [ ] Implement PDF parsing pipelines (GM annual report) leveraging PyPDF + optional LLM verification; log reproducibility metadata.

### 6. Bonus Tracks (Backlog)
- [ ] Credit rating classifier design + data sourcing (e.g., WRDS/SEC filings, rating agency datasets).
- [ ] Risk warning extraction engine leveraging NLP/LLM summarisation; compile list of bankruptcy reports for testing.
- [ ] Loan pricing model survey, dataset curation, and uncertainty quantification workflow.

## Cross-Cutting Considerations
- Maintain modular structure mirroring `mlcoe_q2` for consistency across questions.
- Ensure all scripts accept CLI arguments for tickers/date ranges to ease automation.
- Version control datasets (checksums/metadata) for replicability; store large raw files outside repo if needed and document download scripts.
- Align reporting artifacts with existing conventions (`reports/q2/…`) for unified final submission.
