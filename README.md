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

**Status:** Part 1 implementation complete with data pipelines, model training, and evaluation metrics.

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

# Train forecasting model
python -m mlcoe_q1.pipelines.train_forecaster
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

Interim and final reports are located in:
- `reports/q1/` — Question 1 analysis and results
- `reports/q2/` — Question 2 benchmarks and method comparisons
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
