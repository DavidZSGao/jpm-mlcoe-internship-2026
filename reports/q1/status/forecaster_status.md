# Forecaster Status Report

_Generated: 2025-10-08 09:21:05Z_

## Aggregate Metrics

ticker          mode  observations assets_mae_mean equity_mae_mean identity_gap_mean net_income_mae_mean
  AAPL           mlp             2         $17.74B         $17.52B            $0.00B             $67.98B
   BAC bank_ensemble             2          $0.00B          $0.00B           $-0.00B             $32.17B
     C bank_ensemble             2          $0.00B          $0.00B            $0.00B             $14.07B
   CAT           mlp             2         $10.60B         $14.53B           $-0.00B             $13.20B
    GM           mlp             2         $18.52B         $20.44B            $0.00B             $16.45B
   HON           mlp             2         $10.28B         $10.58B                NA             $11.20B
   JPM bank_ensemble             2          $0.00B          $0.00B            $0.00B             $59.10B
  MSFT           mlp             2         $42.64B         $46.48B            $0.00B             $61.47B
   UNP           mlp             2         $55.50B         $20.28B            $0.00B             $24.45B

## Highlights

- Highest equity MAE tickers: MSFT (mlp): $46.48B, GM (mlp): $20.44B, UNP (mlp): $20.28B
- Highest assets MAE tickers: UNP (mlp): $55.50B, MSFT (mlp): $42.64B, GM (mlp): $18.52B
- Highest net income MAE tickers: AAPL (mlp): $67.98B, MSFT (mlp): $61.47B, JPM (bank_ensemble): $59.10B
- Bank coverage relies on templates for 3 tickers; worst equity MAE is BAC (bank_ensemble): $0.00B.
- Accounting identity gaps remain below $1B across evaluated tickers.
