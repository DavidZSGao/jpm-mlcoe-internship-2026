# Ratio Extraction Log

- `mlcoe_q1/pipelines/extract_pdf_ratios.py` parses GM 2023 10-K, returning values in dollars (scale 1e6). Updated comparison stored at `reports/q1/artifacts/gm_ratio_comparison.parquet`.
- Structured vs PDF differences reflect period mismatch (structured 2024, PDF 2023) and EBIT vs EBITDA proxy choice.
