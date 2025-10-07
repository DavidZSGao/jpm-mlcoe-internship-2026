# Data Ingestion Status — YYYY-MM-DD

- Ingestion utilities for Yahoo Finance are in place (`mlcoe_q1/data/yfinance_ingest.py`).
- Processed parquet transformation pipeline landed (`mlcoe_q1/data/statement_processing.py`) with CLI glue for batch conversion.
- GM, JPM, MSFT, AAPL, HON, CAT, UNP, BAC, and C fundamentals retrieved; processed parquet files live under `mlcoe_q1/data/processed/` with accompanying raw JSON caches.
- Initial yfinance frames were empty due to crumb throttling; fallback now uses Yahoo fundamentals-timeseries with fresh crumb handling. Recent fixes added a DataFrame-to-dict converter so standard yfinance payloads also persist correctly.
- Next action: backfill any remaining prompt tickers (e.g., Tencent ADR) and reconcile period alignment across geographies before scaling experiments.

- Driver feature dataset generated at `mlcoe_q1/data/processed/driver_features.parquet` (log sales, sales-per-asset, growth, margins, leverage ratios).
- Ratio computation CLI: `mlcoe_q1/pipelines/compute_ratios.py` now derives net income and requested leverage/coverage metrics (see `reports/q1/artifacts/gm_ratios.json`).
- PDF parser (`mlcoe_q1/pipelines/extract_pdf_ratios.py`) now reads GM's 10-K tables directly via pdfplumber; outputs stored at `reports/q1/artifacts/gm_pdf_ratios.json`.
