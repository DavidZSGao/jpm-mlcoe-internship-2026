# Question 1 — Balance Sheet Forecasting (Interim Report)

## 1. Executive Summary
- Built an identity-preserving forecasting stack combining deterministic accounting constraints with learned driver dynamics.
- Established an offline-capable data ingestion pipeline (Yahoo fundamentals-timeseries fallback) and processed statements for GM, JPM, MSFT, and AAPL.
- Derived core liquidity/solvency ratios and cost metrics automatically, enabling CFO-facing diagnostics (e.g., GM net income $6.0B, debt-to-equity 2.06×, quick ratio 0.98).
- Baseline persistence backtest achieves single-digit-billion equity mean absolute error for industrial names but struggles with bank-scale balance sheets; initial neural forecaster overestimates assets, highlighting the need for scaling/sector priors.

## 2. Data Pipeline
- CLI workflows: `download_statements`, `prepare_processed_data`, `build_driver_dataset`, `compute_ratios`.
- Artifacts: processed statements (`mlcoe_q1/data/processed/*.parquet`), driver features (`driver_features.parquet`), ratio summaries (`reports/q1/artifacts/gm_ratios.json`).
- Status log maintained at `reports/q1/notes/data_ingestion_status.md`.

## 3. Modelling Components
- **Constraint Layer:** `project_forward` enforces Assets = Liabilities + Equity, tracks financing gaps, and reconciles cash.
- **Driver Features:** Ratios (sales growth, margins, capex/NWC, leverage) computed per ticker for ML consumption.
- **Baselines:** Persistence-based driver replay via `backtest_baseline.py` (MAE examples: GM assets $30.7B, equity $8.7B, cash $5.2B).
- **Neural Forecaster:** Two-layer ReLU MLP trained on driver transitions (`train_forecaster.py`), evaluated end-to-end with constraint projection (`evaluate_forecaster.py`). Initial model overshoots assets (GM MAE ≈ $258B) due to lack of scaling/sector segmentation.

## 4. GM Financial Highlights (FY2024)
- Net income attributable to stockholders: **$6.0B**.
- Cost-to-income ratio: **93.2%**, indicating narrow operating margins.
- Quick ratio: **0.98** — liquidity tight, reliant on inventory conversion.
- Debt metrics: D/E **2.06×**, debt-to-assets **0.46**, debt-to-capital **0.67**, debt-to-EBITDA **5.96×**.
- Interest coverage: **11.1×**; current earnings comfortably service interest, but leverage remains elevated.

### CFO Recommendation
1. **Deleveraging Focus:** Target a debt-to-equity range below 1.5× by allocating a portion of free cash flow to debt reduction; the high debt-to-EBITDA combined with soft margins increases downside risk during EV transition investments.
2. **Working-Capital Discipline:** Quick ratio below 1.0 suggests liquidity pressure — accelerate receivables programmes and review inventory buffers tied to EV launches.
3. **Margin Protection:** With cost-to-income above 90%, pursue additional cost rationalisation (supplier consolidation, manufacturing efficiency) to create headroom for EV R&D without further leverage.

## 5. Work Remaining
- **Driver Modelling:** Normalise features (e.g., log revenues), introduce sector-specific priors, and integrate macro covariates to reduce neural forecast bias.
- **LLM/PDF Extraction:** Automate ratio extraction directly from filings (GM, LVMH, Tencent, etc.) with reproducibility metadata; compare textual extraction to structured pipeline.
- **Ensemble Reporting:** Combine constrained forecasts with LLM commentary for Part 2, produce scenario analysis, and finalise narrative deck.
- **Bonus Tracks:** Credit rating prototype, risk-warning extraction engine, and loan pricing survey pending prioritisation.

## 6. Repository Guide
- Code: `mlcoe_q1/` (data, models, pipelines, utils).
- Reports & notes: `reports/q1/` (artifacts, figures, narrative notes).
- Model outputs: `mlcoe_q1/models/artifacts/driver_forecaster/` (saved weights + history).

