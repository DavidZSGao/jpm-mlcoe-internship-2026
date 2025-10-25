# Bonus — PMMH vs HMC with Differentiable PF

| Sampler | Accept Rate | Chain Length | Notes |
| --- | --- | --- | --- |
| PMMH | 0.25 | 20 | Random-walk MH on $\phi$ (obs noise log-std); likelihood via standard PF |
| HMC | 0.72 | 40 | Uses DPF gradients with entropy-regularised OT resampling |

## Chains (see `reports/artifacts/pmmh_vs_hmc.json`)
- PMMH samples drift slowly with larger variance (RW proposal std 0.12).
- HMC samples contract around -0.05 with smaller variance and faster mixing.

## Reproduce

```bash
python -m mlcoe_q2.pipelines.pmmh_vs_hmc_dpf \
  --config configs/q2/pmmh_vs_hmc_dpf.json
```

> Tip: the config enables `--eager` to cut TensorFlow compilation overhead; runs still take a few minutes on CPU because the HMC kernel differentiates through the differentiable particle filter.
