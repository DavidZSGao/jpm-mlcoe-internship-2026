# Forecaster Evaluation Snapshot

The latest driver forecaster (trained with the bank-aware head) was backtested using the refreshed templates and summarised with `python -m mlcoe_q1.pipelines.report_forecaster_status --evaluation reports/q1/artifacts/forecaster_eval.parquet`.

| Ticker | Mode | Observations | Assets MAE (USD B) | Equity MAE (USD B) | Identity Gap (USD B) |
| --- | --- | --- | --- | --- | --- |
| AAPL | mlp | 2 | 3.58 | 17.90 | 0.00 |
| BAC | bank_template | 2 | 106.40 | 383.35 | 0.00 |
| C | bank_template | 2 | 19.19 | 355.92 | 0.00 |
| CAT | mlp | 2 | 11.33 | 7.26 | 0.00 |
| GM | mlp | 2 | 41.84 | 7.03 | 0.00 |
| HON | mlp | 2 | 19.44 | 3.12 | NA |
| JPM | bank_template | 2 | 144.24 | 606.87 | 0.00 |
| MSFT | mlp | 2 | 27.83 | 64.07 | 0.00 |
| UNP | mlp | 2 | 29.33 | 0.82 | 9.88 |

Key takeaways:

- Bank templates continue to dominate equity error, with JPM at roughly $607B MAE and BAC/C trailing closely; reducing these requires richer bank-specific drivers and potentially sequence models.
- Accounting identity gaps remain tightly controlled (≤$1B) except for UNP, where the gap spikes to ~$9.9B because the evaluation sample mixes template and neural outputs across sparse filings—this is the next candidate for feature review.
- Non-bank tickers fall below ~$65B equity MAE, validating that the shared MLP head remains competitive when the driver dataset is dense.
