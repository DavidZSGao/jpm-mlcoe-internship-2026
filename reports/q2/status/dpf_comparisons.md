# Q2 DPF Resampling Comparisons

| Method | Runtime (s) | Peak Mem (KB) | LogLik | Mean ESS |
| --- | --- | --- | --- | --- |
| DPF_Soft | 61.68 ± 10.88 | 33580.40 ± 322.49 | -87.56 ± 0.89 | 120.03 ± 16.18 |
| DPF_OT_Low | 60.81 ± 5.35 | 33514.01 ± 10.11 | -97.63 ± 11.31 | 163.17 ± 36.44 |
| DPF_OT | 58.57 ± 5.51 | 34029.72 ± 158.29 | -103.64 ± 15.79 | 181.47 ± 23.36 |

## Notes
- Aggregated across multiple seeds
- Methods: soft weights (no transport), OT low-iter, and full OT

## Reproduce

```bash
python -m mlcoe_q2.pipelines.run_dpf_comparisons_multiseed \
  --config configs/q2/run_dpf_comparisons_multiseed.json
```
