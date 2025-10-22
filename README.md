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

**Status:** Part 1 and Part 2 are now fully delivered. The Strategic Lending workflow covers deterministic projections, probabilistic TensorFlow forecasters (MLP/GRU/variational/Gaussian heads), Monte Carlo scenario packaging, macro overlays, Altman credit scoring, loan pricing, LLM benchmarking (HF + hosted APIs, multi-seed variance), risk-warning extraction, governance audits, and the end-to-end `publish_lending_submission` hand-off. Roadmap notes in `reports/q1/q1_workplan.md` now focus on operational governance follow-ups rather than missing functionality; analysts looking for a step-by-step operational guide can start with `reports/q1/strategic_lending_playbook.md`.

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
# ...or reuse the repository preset
python -m mlcoe_q1.pipelines.build_driver_dataset --config configs/q1/build_driver_dataset.json

# Train forecasting model and calibrate bank ensembles
python -m mlcoe_q1.pipelines.train_forecaster --processed-root mlcoe_q1/data/processed \
    --drivers mlcoe_q1/data/processed/driver_features.parquet --calibrate-banks
# ...or reuse the repository preset
python -m mlcoe_q1.pipelines.train_forecaster --config configs/q1/train_forecaster.json

# Evaluate forecasts (auto-selects the calibrated bank ensemble when available)
python -m mlcoe_q1.pipelines.evaluate_forecaster --processed-root mlcoe_q1/data/processed \
    --drivers mlcoe_q1/data/processed/driver_features.parquet \
    --model-dir mlcoe_q1/models/artifacts/driver_forecaster \
    --output reports/q1/artifacts/forecaster_eval.parquet
# ...or reuse the repository preset
python -m mlcoe_q1.pipelines.evaluate_forecaster --config configs/q1/evaluate_forecaster.json

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

# (Re)use JSON/YAML presets when calling the adapter or benchmark suite
python -m mlcoe_q1.pipelines.run_llm_adapter \
    --config configs/q1/run_llm_adapter.json
python -m mlcoe_q1.pipelines.benchmark_llm_suite \
    --config configs/q1/benchmark_llm_suite.json

# Reuse presets for scenario/macro/reporting workflows
python -m mlcoe_q1.pipelines.package_scenarios \
    --config configs/q1/package_scenarios.json
python -m mlcoe_q1.pipelines.assess_scenario_reasonableness \
    --config configs/q1/assess_scenario_reasonableness.json
python -m mlcoe_q1.pipelines.simulate_macro_conditions \
    --config configs/q1/simulate_macro_conditions.json
python -m mlcoe_q1.pipelines.analyze_forecaster_calibration \
    --config configs/q1/analyze_forecaster_calibration.json
python -m mlcoe_q1.pipelines.generate_lending_briefing \
    --config configs/q1/generate_lending_briefing.json
python -m mlcoe_q1.pipelines.generate_executive_summary \
    --config configs/q1/generate_executive_summary.json
python -m mlcoe_q1.pipelines.compile_lending_package \
    --config configs/q1/compile_lending_package.json
python -m mlcoe_q1.pipelines.price_loans \
    --config configs/q1/price_loans.json
python -m mlcoe_q1.pipelines.generate_cfo_recommendations \
    --config configs/q1/generate_cfo_recommendations.json

# Generate a markdown + JSON progress digest of the Question 1 workplan
python -m mlcoe_q1.pipelines.summarize_workplan_progress \
    --config configs/q1/summarize_workplan_progress.json

# Turn the benchmark manifest into ranked tables and a Markdown briefing
python -m mlcoe_q1.pipelines.summarize_llm_benchmarks \
    --manifest reports/q1/artifacts/llm_runs/manifest.json \
    --output reports/q1/artifacts/llm_runs/benchmark_ranked.parquet \
    --markdown-output reports/q1/status/llm_benchmark_summary.md

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

# Run the full Strategic Lending refresh (summaries → scenarios → macro overlays → briefings → packaging)
python -m mlcoe_q1.pipelines.orchestrate_lending_workflow \
    --evaluation reports/q1/artifacts/forecaster_eval.parquet \
    --summary-output reports/q1/status/forecaster_summary.parquet \
    --scenario-output reports/q1/artifacts/forecaster_scenarios.parquet \
    --macro-output reports/q1/artifacts/macro_conditioned_scenarios.parquet \
    --briefing-output reports/q1/status/strategic_lending_briefing.md

# The orchestrator and publisher both accept --config for reusable JSON/YAML defaults
python -m mlcoe_q1.pipelines.orchestrate_lending_workflow \
    --config configs/q1/orchestrate_lending_workflow.json

# Publish a governance-ready package (orchestrator + executive summary + audit + optional zip)
python -m mlcoe_q1.pipelines.publish_lending_submission \
    --evaluation reports/q1/artifacts/forecaster_eval.parquet \
    --prompt-dataset reports/q1/artifacts/llm_prompts.parquet \
    --responses-root reports/q1/artifacts/llm_runs \
    --deliverable-root reports/q1/deliverables/strategic_lending \
    --zip-output reports/q1/deliverables/strategic_lending.zip

# Example using the publishing config (includes audit expectations and bundle paths)
python -m mlcoe_q1.pipelines.publish_lending_submission \
    --config configs/q1/publish_lending_submission.json
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
