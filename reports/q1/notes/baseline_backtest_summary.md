# Baseline Backtest Summary

- Naive driver persistence (carry forward last observed margins/growth) evaluated via `project_forward`.
- Large absolute errors for bank balance sheets (JPM) due to leverage ratios forcing debt expansion; need sector-specific calibration.
- Industrial/tech names (GM, MSFT, AAPL) show asset MAE in the $10-35B range with cash errors more pronounced when cash policies shift.
- Next: learn driver forecasts (AR models or small neural nets) and calibrate leverage/tax assumptions per sector.
## Neural Forecaster Evaluation
- Trained `tf_forecaster.py` MLP (2x64 relu) on driver transitions across GM/JPM/MSFT/AAPL.
- Evaluation stored at `reports/q1/artifacts/forecaster_evaluation.parquet`. Current model over-predicts balance sheets (asset MAE 250B+), highlighting need for scaling/sector conditioning before deployment.
- Next iterations: normalise features (log-scaling for sales), include macro covariates, and constrain forecasted leverage/payout via clipping or projection to realistic bands.
- Normalised MLP (log/standardized features + sector flag) reduces industrial asset MAE to ~\$28–$27B (AAPL/GM) versus persistence baseline but banks remain challenging (JPM ~\$827B MAE). Further tuning: separate bank model, incorporate macro drivers, widen training corpus.
- Applying log1p transform to sales improved MAE further (GM ≈ $25.3B, AAPL/MSFT ≈ $25–26B), but JPM remains elevated; plan to train dedicated bank forecaster or revert to persistence for financials.
- Banks currently evaluated via persistence fallback (mode recorded in `forecaster_evaluation.parquet`). JPM MAE remains ≈ $8.27e11, highlighting need for bank-specific state dynamics beyond our driver set.
- Verified GM ratios by parsing the PDF directly (`extract_pdf_ratios.py`); net income $10.1B, cost/income 94.6%, quick ratio 0.90, debt-to-equity 1.89×.
- Comparison between structured dataset and PDF extraction stored at `reports/q1/artifacts/gm_ratio_comparison.parquet`; note PDF presents figures in millions (net income 10,127 vs structured $6.01B), so we need scaling alignment before blending sources.
