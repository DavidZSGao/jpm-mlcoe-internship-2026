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
- `python -m mlcoe_q1.pipelines.build_driver_dataset [--tickers ... --lags 2 --lag-features sales sales_growth]` derives forecasting driver ratios with log-revenue and per-asset normalisation, now including tangible-equity, interest-spread, and year-over-year growth metrics for banks and corporates, and can append lagged histories before writing `driver_features.parquet`.
- `python -m mlcoe_q1.pipelines.train_forecaster` trains an MLP on driver progression with a bank-aware output head, exports bank templates from `data/processed/`, and saves artifacts under `models/artifacts/` (override data location with `--processed-root`). Pass `--calibrate-banks` to fit ensemble weights after training.
- `python -m mlcoe_q1.pipelines.evaluate_forecaster --bank-mode {auto,template,mlp,persistence,ensemble}` backtests the saved model; `auto` prefers the calibrated ensemble when available and falls back to template/MLP/persistence strategies as needed.
- `python -m mlcoe_q1.pipelines.calibrate_bank_ensemble` recomputes the linear ensemble weights that blend bank templates with neural projections, updating `bank_ensemble.json` alongside the model artifacts.
- `python -m mlcoe_q1.pipelines.summarize_forecaster_eval [--group-by ticker mode]` aggregates the evaluation parquet into grouped MAE/identity-gap statistics (including net income MAE when present) and can persist CSV/JSON/Parquet summaries with `--output`.
- `python -m mlcoe_q1.pipelines.report_forecaster_status [--group-by ticker mode]` renders the summarised MAE/identity-gap/net-income metrics into a Markdown status report, optionally writing to disk via `--output`.
- `python -m mlcoe_q1.pipelines.build_llm_prompt_dataset [--statements balance_sheet income_statement]` creates prompt/label pairs for Part 2 LLM experiments by pairing prior-period statements with ground-truth targets; persistable via `--output`.
- `python -m mlcoe_q1.pipelines.run_llm_adapter --prompts ... --adapter flan-t5 --model t5-small` invokes a HuggingFace text-to-text model to generate JSON forecasts for the prompt corpus with automatic truncation of overlong inputs; use `--limit` for sampling smaller batches during development.
- `python -m mlcoe_q1.pipelines.generate_llm_baseline_responses --prompt-dataset ... --strategy {context_copy,scaled_copy}` still ships as a deterministic baseline for comparison.
- `python -m mlcoe_q1.pipelines.evaluate_llm_responses --prompt-dataset ... --responses ...` scores JSON responses from LLMs against the prompt dataset, computing coverage, MAE/MAPE, and exportable per-record metrics with `--output`.
- `python -m mlcoe_q1.pipelines.compare_llm_and_forecaster --forecaster-eval ... --llm-metrics ...` joins the structured evaluation artifact with LLM metrics on matching ticker/period slices, prints grouped summaries (default by ticker), and can persist both the merged table (`--output`) and aggregate statistics (`--summary-output`).
- `python -m mlcoe_q1.pipelines.generate_cfo_recommendations --forecaster-eval ... --llm-eval ...` renders Markdown guidance for CFOs/CEOs by combining structured MAE metrics with LLM coverage diagnostics.
- `python -m mlcoe_q1.pipelines.validate_driver_dataset [--min-observations 3 --max-gap-days 500]` inspects the driver dataset for duplicate periods, feature gaps, long filing intervals, and can persist the validation report with `--output`.
- `python -m mlcoe_q1.pipelines.extract_pdf_ratios --company {gm,lvmh,tencent} PATH/TO/REPORT.pdf` extracts ratios directly from the PDF and records provenance metadata; pass `--config` with a JSON override to support additional layouts.

## Notes
- The downloader falls back to Yahoo's fundamentals-timeseries API when `yfinance` yields empty frames, so direct HTTP access is required on first fetch.
- `mlcoe_q1/models/balance_sheet_constraints.py` provides deterministic identity-preserving projections that map driver vectors to full statements.
- Processed data and drivers currently cover a nine-ticker mix of cyclicals/industrials (GM, HON, CAT, UNP), large-cap comparables (AAPL, MSFT), and banks (JPM, BAC, C) for sector-aware training experiments, with bank rows now capturing net-interest, gross-interest, and tangible-equity ratios alongside the shared corporate features.
- The TensorFlow forecaster blends a shared corporate head with a dedicated bank head controlled by the `is_bank` auxiliary feature so financial institutions can specialise without retraining a separate network.
- See `reports/q1/deterministic_balance_sheet_spec.md` for the mathematical walkthrough of the projection engine and its simulation framing.
- Earnings coverage and next steps are documented in `reports/q1/notes/earnings_linkage.md`.
- Model extension priorities and probabilistic roadmap live in `reports/q1/notes/ml_extension_roadmap.md`.
- Simulation framing and exogenous driver assumptions are detailed in `reports/q1/notes/simulation_strategy.md`.
- Literature highlights from Vélez-Pareja, Mejía-Pelaez, Shahnazarian, and Samonas are summarised in `reports/q1/literature_summary.md` with implementation takeaways for this codebase.
- HuggingFace adapters rely on the CPU `torch` build plus `transformers`/`sentencepiece`; install via `pip install -r requirements.txt` and expect the first inference run to download ~200 MB of weights.
- The PDF ratio extractor depends on `pdfplumber>=0.11.0`; install with `pip install -r requirements.txt` before running the CLI.
