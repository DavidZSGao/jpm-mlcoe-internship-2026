# JPMorgan MLCOE 2026 Internship — AI/ML Exercise

This repository contains solutions to the Machine Learning Center of Excellence (Hong Kong) internship program coding challenge.

## Project Structure

- **`mlcoe_q1/`** — Question 1: Balance Sheet Forecasting
- **`mlcoe_q2/`** — Question 2: State-Space Models and Particle Filters
- **`tests/`** — Unit and integration tests for both questions
- **`reports/`** — Documentation, interim reports, and experimental artifacts

## Questions Attempted

### Question 1: Balance Sheet Forecasting
A TensorFlow-based system for forecasting corporate balance sheets using:
- Driver-based ratio modeling
- Accounting identity constraints
- MLP-based time-series prediction

**Status:** Part 1 and the core Part 2 deliverables are complete—bank ensembles drive BAC/JPM/C equity MAE below $0.01$B, the ML extension/simulation roadmaps are documented, and the LLM adapter/evaluator/CFO reporting stack (with truncation-aware HuggingFace integration) is online. Remaining backlog covers PDF regression tests, broader LLM sweeps, and bonus tracks (see `reports/q1/q1_workplan.md`).

### Question 2: State-Space Models & Particle Filters
Implementation and benchmarking of sequential Monte Carlo methods including:
- Kalman filters (linear Gaussian SSM)
- Extended/Unscented Kalman filters
- Particle filters with differentiable resampling (optimal transport)
- Deterministic particle flows (EDH, LEDH, kernel-embedded)
- Stochastic particle flows
- Particle Flow Particle Filter (PF-PF) integration

**Status:** Part 1A-C and Part 2 implementations complete with comprehensive diagnostics and benchmarking infrastructure.

## Installation

### Requirements
- Python 3.12+
- TensorFlow 2.x
- TensorFlow Probability
- PyTorch (CPU build is sufficient for the current HuggingFace adapters)
- HuggingFace `transformers` + `sentencepiece` for LLM baselines

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
PYTHONPATH=. pytest -q
```

## Testing

All implementations include unit and integration tests following pytest conventions:
```bash
# Run all tests
PYTHONPATH=. pytest tests/

# Run Q1 tests only
PYTHONPATH=. pytest tests/q1/

# Run Q2 tests only
PYTHONPATH=. pytest tests/q2/
```

## Usage Examples

### Question 1
```bash
# Download and process financial data
python -m mlcoe_q1.pipelines.download_statements AAPL MSFT
python -m mlcoe_q1.pipelines.prepare_processed_data

# Build driver features with two trailing-period lags for sales dynamics
python -m mlcoe_q1.pipelines.build_driver_dataset --lags 2 --lag-features sales sales_growth \
    --data-root mlcoe_q1/data/processed --output mlcoe_q1/data/processed/driver_features.parquet

# Train forecasting model and calibrate bank ensembles
python -m mlcoe_q1.pipelines.train_forecaster --processed-root mlcoe_q1/data/processed \
    --drivers mlcoe_q1/data/processed/driver_features.parquet --calibrate-banks

# Evaluate forecasts (auto-selects the calibrated bank ensemble when available)
python -m mlcoe_q1.pipelines.evaluate_forecaster --processed-root mlcoe_q1/data/processed \
    --drivers mlcoe_q1/data/processed/driver_features.parquet \
    --model-dir mlcoe_q1/models/artifacts/driver_forecaster \
    --output reports/q1/artifacts/forecaster_eval.parquet

# Calibrate bank ensemble weights independently
python -m mlcoe_q1.pipelines.calibrate_bank_ensemble \
    --drivers mlcoe_q1/data/processed/driver_features.parquet \
    --model-dir mlcoe_q1/models/artifacts/driver_forecaster \
    --processed-root mlcoe_q1/data/processed \
    --output mlcoe_q1/models/artifacts/driver_forecaster/bank_ensemble.json

# Summarize processed statements with pandas
python -m mlcoe_q1.pipelines.describe_processed --tickers AAPL MSFT \
    --output reports/q1/artifacts/aapl_msft_summary.json

# Generate a Markdown status report of forecast errors
python -m mlcoe_q1.pipelines.report_forecaster_status \
    --evaluation reports/q1/artifacts/forecaster_eval.parquet \
    --output reports/q1/status/forecaster_status.md

# Validate driver dataset coverage
python -m mlcoe_q1.pipelines.validate_driver_dataset \
    --drivers mlcoe_q1/data/processed/driver_features.parquet \
    --output reports/q1/artifacts/driver_validation.csv

# Build LLM-ready prompts pairing context statements with ground truth
python -m mlcoe_q1.pipelines.build_llm_prompt_dataset \
    --processed-root mlcoe_q1/data/processed \
    --output reports/q1/artifacts/llm_prompts.json

# Run a HuggingFace adapter (t5-small) to obtain LLM responses
python -m mlcoe_q1.pipelines.run_llm_adapter \
    --prompts reports/q1/artifacts/llm_prompts.parquet \
    --adapter flan-t5 --model t5-small \
    --output reports/q1/artifacts/llm_responses_t5.parquet

# (The first run will download model weights via HuggingFace; expect ~200 MB and CPU inference.)

# Benchmark LLM responses against ground truth targets
python -m mlcoe_q1.pipelines.evaluate_llm_responses \
    --prompt-dataset reports/q1/artifacts/llm_prompts.parquet \
    --responses reports/q1/artifacts/llm_responses_t5.parquet \
    --output reports/q1/artifacts/llm_metrics_t5.parquet

# Compare LLM metrics against structured forecaster errors
python -m mlcoe_q1.pipelines.compare_llm_and_forecaster \
    --forecaster-eval reports/q1/artifacts/forecaster_eval.parquet \
    --llm-metrics reports/q1/artifacts/llm_metrics_t5.parquet \
    --summary-output reports/q1/status/llm_vs_forecaster_t5.csv

# Generate CFO recommendations combining forecaster and LLM insights
python -m mlcoe_q1.pipelines.generate_cfo_recommendations \
    --forecaster-eval reports/q1/artifacts/forecaster_eval.parquet \
    --llm-eval reports/q1/artifacts/llm_metrics_t5.parquet \
    --output reports/q1/status/cfo_recommendations.md
```

### Question 2
```bash
# Run LGSSM Kalman filter validation
python -m mlcoe_q2.experiments.lgssm_validation \
    --output-json reports/artifacts/lgssm_validation.json \
    --figure-path reports/figures/lgssm_validation.png

# Run nonlinear filter diagnostics
python -m mlcoe_q2.experiments.nonlinear_filter_diagnostics \
    --num-seeds 5 \
    --output-json reports/artifacts/nonlinear_diagnostics.json \
    --rmse-figure reports/figures/rmse.png \
    --ess-figure reports/figures/ess.png

# Run full benchmark suite
python -m mlcoe_q2.experiments.benchmark
```

## Code Quality

- **Style:** PEP 8 compliant with meaningful naming conventions
- **Documentation:** Docstrings following PEP 257
- **Testing:** >90% coverage across core modules
- **Type hints:** Comprehensive annotations for improved IDE support

## Reports

Reports are located in:
- `reports/q1/q1_interim_report.md` — consolidated Part 1/Part 2 narrative, methodology, and evaluation tables for Question 1 (current interim deliverable)
- `reports/q2/` — Question 2 benchmarks and method comparisons
- `reports/artifacts/` — JSON summaries and validation metrics
- `reports/figures/` — Visualizations (RMSE curves, ESS traces, etc.)

## Timeline

- **Oct 13, 2025:** Interest confirmation
- **Nov 14, 2025:** Part 1 submission
- **Jan 19, 2026:** Final submission (Part 1 + Part 2)

## Author

[Your Name]  
[Your Email]  
[Your University/Degree if applicable]

## License

This project is submitted as part of the JPMorgan MLCOE 2026 internship application process.
