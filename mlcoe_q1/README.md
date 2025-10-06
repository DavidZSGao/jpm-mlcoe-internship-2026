# Question 1 — Balance Sheet Forecasting

This package contains code and assets for the lending application prototype described in Question 1.

## Layout
- `data/` — raw downloads and intermediate feature sets for financial statements.
- `models/` — TensorFlow models and supporting modules for balance-sheet simulation.
- `pipelines/` — end-to-end training and evaluation workflows.
- `experiments/` — scripts and notebooks for reproducible experimentation.
- `evaluation/` — metrics, diagnostics, and reporting utilities.
- `utils/` — shared helpers (I/O, accounting identities, feature engineering).

Tests for this module live under `tests/q1`.

## Usage
- `python -m mlcoe_q1.pipelines.download_statements TICKER ...` downloads or loads cached statements into `data/raw/` (pass `--cache-only` to force offline mode).
- `python -m mlcoe_q1.pipelines.prepare_processed_data [TICKER ...]` converts cached bundles into parquet tables under `data/processed/`.
- `python -m mlcoe_q1.pipelines.build_driver_dataset [--tickers ...]` derives forecasting driver ratios and writes `driver_features.parquet`.
- `python -m mlcoe_q1.pipelines.train_forecaster` trains an MLP on driver progression and saves weights under `models/artifacts/`.

## Notes
- The downloader falls back to Yahoo's fundamentals-timeseries API when `yfinance` yields empty frames, so direct HTTP access is required on first fetch.
- `mlcoe_q1/models/balance_sheet_constraints.py` provides deterministic identity-preserving projections that map driver vectors to full statements.
