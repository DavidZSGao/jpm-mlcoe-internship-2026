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
- `python -m mlcoe_q1.pipelines.build_driver_dataset [--tickers ...]` derives forecasting driver ratios with log-revenue and per-asset normalisation and writes `driver_features.parquet`.
- `python -m mlcoe_q1.pipelines.train_forecaster` trains an MLP on driver progression, exports bank templates from `data/processed/`, and saves artifacts under `models/artifacts/` (override data location with `--processed-root`).
- `python -m mlcoe_q1.pipelines.evaluate_forecaster --bank-mode {auto,template,mlp,persistence}` backtests the saved model with optional bank strategies (template default, MLP-only, or persistence fallback) and writes a parquet report of MAE metrics.
- `python -m mlcoe_q1.pipelines.summarize_forecaster_eval [--group-by ticker mode]` aggregates the evaluation parquet into grouped MAE/identity-gap statistics and can persist CSV/JSON/Parquet summaries with `--output`.
- `python -m mlcoe_q1.pipelines.extract_pdf_ratios --company {gm,lvmh,tencent} PATH/TO/REPORT.pdf` extracts ratios directly from the PDF and records provenance metadata; pass `--config` with a JSON override to support additional layouts.

## Notes
- The downloader falls back to Yahoo's fundamentals-timeseries API when `yfinance` yields empty frames, so direct HTTP access is required on first fetch.
- `mlcoe_q1/models/balance_sheet_constraints.py` provides deterministic identity-preserving projections that map driver vectors to full statements.
- Processed data and drivers currently cover a nine-ticker mix of cyclicals/industrials (GM, HON, CAT, UNP), large-cap comparables (AAPL, MSFT), and banks (JPM, BAC, C) for sector-aware training experiments.
- See `reports/q1/deterministic_balance_sheet_spec.md` for the mathematical walkthrough of the projection engine and its simulation framing.
- Literature highlights from Vélez-Pareja, Mejía-Pelaez, Shahnazarian, and Samonas are summarised in `reports/q1/literature_summary.md` with implementation takeaways for this codebase.
- The PDF ratio extractor depends on `pdfplumber>=0.11.0`; install with `pip install -r requirements.txt` before running the CLI.
