# Q2 Filter Benchmark Status

| Method | Runtime (s) | Peak Mem (KB) | LogLik | Mean ESS |
| --- | --- | --- | --- | --- |
| PF | 52.18 ± 1.94 | 33565.63 ± 326.51 | -30.33 ± 14.50 | 169.56 ± 10.21 |
| DifferentiablePF | 51.68 ± 1.79 | 34056.45 ± 190.05 | -103.64 ± 15.79 | 181.47 ± 23.36 |
| EKF | 10.61 ± 1.66 | 2414.06 ± 89.65 | -10.09 ± 0.72 | — |
| UKF | 1.84 ± 0.10 | 2206.99 ± 14.51 | -9.99 ± 0.90 | — |

## Notes
- **seeds**: aggregated across multiple seeds
- **DPF** uses OT resampling; hyperparameters fixed for comparability

## Reproduce

```bash
python -m mlcoe_q2.pipelines.run_multiseed_benchmarks \
  --config configs/q2/run_multiseed_benchmarks.json
```
