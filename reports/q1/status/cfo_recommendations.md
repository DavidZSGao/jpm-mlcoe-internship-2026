# CFO Recommendations

## Forecast Accuracy Snapshot
| ticker | observations | assets_mae_mean | equity_mae_mean | net_income_mae_mean |
| --- | --- | --- | --- | --- |
| MSFT | 2 | 4.26e+10 | 4.65e+10 | 6.15e+10 |
| GM | 2 | 1.85e+10 | 2.04e+10 | 1.64e+10 |
| UNP | 2 | 5.55e+10 | 2.03e+10 | 2.45e+10 |
| AAPL | 2 | 1.77e+10 | 1.75e+10 | 6.80e+10 |
| CAT | 2 | 1.06e+10 | 1.45e+10 | 1.32e+10 |
| HON | 2 | 1.03e+10 | 1.06e+10 | 1.12e+10 |
| BAC | 2 | 5.37e+03 | 9.58e+03 | 3.22e+10 |
| JPM | 2 | 6.72e+04 | 1.38e+03 | 5.91e+10 |
| C | 2 | 1.88e+03 | 9.26e+02 | 1.41e+10 |

## Ticker-Level Guidance
- **MSFT** — Equity forecasts are stable; maintain current underwriting thresholds. Earnings volatility is material; stress-test cash flow coverage scenarios.
- **GM** — Equity forecasts are stable; maintain current underwriting thresholds. Earnings volatility is material; stress-test cash flow coverage scenarios.
- **UNP** — Equity forecasts are stable; maintain current underwriting thresholds. Earnings volatility is material; stress-test cash flow coverage scenarios.

## LLM Diagnostic
| ticker | mae_mean | coverage_mean | invalid_mean |
| --- | --- | --- | --- |
| AAPL | N/A | 0.00e+00 | 0.00e+00 |
| BAC | N/A | 0.00e+00 | 0.00e+00 |

- **AAPL** — Low coverage — rely on deterministic model for critical ratios.
- **BAC** — Low coverage — rely on deterministic model for critical ratios.

## Next Steps
Focus on lowering residual bank MAE through driver enrichment and leverage audits while using the LLM runs for qualitative sanity checks rather than hard targets.
