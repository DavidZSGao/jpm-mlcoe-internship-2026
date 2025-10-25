# Q2 PF-PF (Dai 2022) Parameter Sweep

| Method | Runtime (s) | Peak Mem (KB) | LogLik | Mean ESS |
| --- | --- | --- | --- | --- |
| PF_PF_LEDH | 26081.61 ± 1079.88 | 146101.63 ± 520.48 | -17.81 ± 6.26 | 179.18 ± 5.78 |
| PF_PF_SPF_A | 31533.59 ± 6539.03 | 302532.32 ± 330.97 | -15.49 ± 8.14 | 174.17 ± 9.49 |
| PF_PF_SPF_B | 24244.14 ± 1420.53 | 302579.65 ± 138.22 | -13.02 ± 4.67 | 177.55 ± 10.36 |
| PF_PF_SPF_C | 40650.75 ± 4895.40 | 429146.43 ± 361.17 | -17.19 ± 3.13 | 177.70 ± 8.62 |

## Notes
- Aggregated across multiple seeds
- Stochastic Particle Flow configurations approximate Dai (2022): smaller step-size, more steps, variable diffusion

## Reproduce

```bash
python -m mlcoe_q2.pipelines.run_pfpf_dai22_multiseed \
  --config configs/q2/run_pfpf_dai22_multiseed.json
```
