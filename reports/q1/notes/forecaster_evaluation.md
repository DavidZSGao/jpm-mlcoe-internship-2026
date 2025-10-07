# Forecaster Evaluation Snapshot

The latest driver forecaster (trained with bank tickers included) was backtested using the updated templates.

| Ticker | Assets MAE (USD) | Equity MAE (USD) | Mode |
| --- | --- | --- | --- |
| AAPL | 3.58e9 | 1.79e10 | mlp |
| BAC | 1.06e11 | 3.83e11 | bank_template |
| C | 1.92e10 | 3.56e11 | bank_template |
| CAT | 1.13e10 | 7.26e9 | mlp |
| GM | 4.18e10 | 7.03e9 | mlp |
| HON | 1.94e10 | 3.12e9 | mlp |
| JPM | 1.44e11 | 6.07e11 | bank_template |
| MSFT | 2.78e10 | 6.41e10 | mlp |
| UNP | 2.93e10 | 8.18e8 | mlp |

The bank templates now pull the latest liability ratio, dropping JPM's equity MAE to roughly 6×10^11 from the previous ~8×10^11 baseline.
