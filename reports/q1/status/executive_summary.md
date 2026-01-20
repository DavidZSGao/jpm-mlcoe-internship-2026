# Strategic Lending Executive Summary

## Forecast quality
- Assets MAE: 44,798,003,786.45
- Equity MAE: 160,704,983,952.38
- Identity gap: 1,235,499,707.71
- Lowest assets MAE: AAPL (3,583,806,114.32)
- Highest assets MAE: JPM (144,243,296,971.10)

## Scenario diagnostics
- **baseline**; Assets MAE: 44,798,003,786.45; Assets MAPE: 0.13; Equity MAE: 160,704,983,952.38; Equity MAPE: 0.72; Net income MAE: N/A; Net income MAPE: N/A; Identity-gap MAE: 1,235,499,707.71; n=18
- **downside**; Assets MAE: 44,798,003,786.45; Assets MAPE: 0.13; Equity MAE: 160,704,983,952.38; Equity MAPE: 0.72; Net income MAE: N/A; Net income MAPE: N/A; Identity-gap MAE: 1,235,499,707.71; n=18
- **upside**; Assets MAE: 44,798,003,786.45; Assets MAPE: 0.13; Equity MAE: 160,704,983,952.38; Equity MAPE: 0.72; Net income MAE: N/A; Net income MAPE: N/A; Identity-gap MAE: 1,235,499,707.71; n=18

## Calibration
- Calibration report not found or empty.

## Macro overlays
- **baseline**; macro: gdp_growth=0.018, policy_rate=0.0425, unemployment_rate=0.037; adjustments: 0
- **mild_downturn**; macro: gdp_growth=0.003, policy_rate=0.038, unemployment_rate=0.049; adjustments: 2 (identity_gap, pred_equity)
- **rate_shock**; macro: gdp_growth=0.01, policy_rate=0.0625, unemployment_rate=0.042; adjustments: 2 (pred_equity, pred_total_assets)

## LLM benchmarking
- openai-chat/gpt-4o-mini: MAE 9,946,035,143.21 (± 21,525,185,369.32); coverage 81.3% (± 37.7%); records=99; seeds=3

## Loan pricing
- **baseline**; avg rate: 8.7%; avg spread: 4.2%; n=18
- **optimistic**; avg rate: 7.1%; avg spread: 2.6%; n=18
- **stress**; avg rate: 10.3%; avg spread: 5.8%; n=18

## Credit analytics
- Tickers covered: 3333.HK, AAPL, CAT, GM, HON, MSFT, UNP
- Period range: 2020-12-31T00:00:00 → 2025-06-30T00:00:00
- Observations: 27

## Risk warnings
- BBBY: 26 flagged warnings
- Sears: 6 flagged warnings
