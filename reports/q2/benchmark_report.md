# Question 2 Benchmark Summary

## Overview
This report captures the consolidated benchmarking results for the Question&nbsp;2 filtering and particle-flow methods. Measurements were generated with `mlcoe_q2/pipelines/benchmark.py` (TensorFlow backend, seed `0`, 15-step nonlinear SSM scenario).

## Experimental Setup
- **Model**: Nonlinear SSM defined in `_build_nonlinear_model()` within `mlcoe_q2/pipelines/benchmark.py`.
- **Sequence length**: 15 observation steps simulated once per run.
- **Particles**: 256 for `particle_filter()` and `differentiable_particle_filter()`; deterministic flows reuse the same count.
- **Hardware**: CPU execution (TensorFlow retracing warnings observed; see notes).

## Filtering Methods
| Method | Runtime (s) | Peak Memory (KB) | Log-Likelihood | Mean ESS |
| --- | ---: | ---: | ---: | ---: |
| PF | 50.83 | 34,117.58 | -17.74 | 174.61 |
| Differentiable PF | 53.13 | 34,162.87 | -91.28 | 209.31 |
| EKF | 9.86 | 2,535.95 | -9.44 | — |
| UKF | 1.75 | 2,269.09 | -9.28 | — |

**Highlights**
- **Likelihood**: EKF/UKF deliver higher (less negative) log-likelihoods than particle approaches in this configuration.
- **ESS**: Differentiable PF increases mean ESS by ~20% versus SIR PF, despite worse likelihood—indicates smoother weight distribution but possible model mismatch.
- **Runtime/Memory**: PF variants dominate compute time and memory owing to TensorFlow OT resampling; deterministic filters remain lightweight.

## Deterministic Particle Flows
| Flow | Total Runtime (s) | Peak Memory (KB) | Mean Particle Movement | Mean Residual (Before) | Mean Residual (After) |
| --- | ---: | ---: | ---: | ---: | ---: |
| EDH | 5,529.07 | 18,490.09 | 0.48 | 0.34 | 0.12 |
| LEDH | 4,192.26 | 17,444.06 | 0.22 | 0.37 | 0.26 |
| Kernel | 84.36 | 3,713.99 | 0.79 | 0.35 | 0.09 |

**Highlights**
- **Kernel flow** achieves the largest reduction in residual norm with moderate runtime and memory.
- **EDH/LEDH** exhibit great runtime variance—profiling indicates costly Jacobian computation per step (thousands of seconds on CPU).
- Movement metrics align with expected aggressiveness (Kernel > EDH > LEDH).

## Nonlinear Filter Diagnostics (Multi-Seed)
- Aggregated diagnostics live in `reports/artifacts/nonlinear_diagnostics.json` (generated via `mlcoe_q2.pipelines.nonlinear_filter_diagnostics`).
- RMSE/ESS plots land at `reports/figures/nonlinear_rmse.png` and `reports/figures/nonlinear_ess.png`.
- **EKF**: linearisation bias accumulates for the sine-heavy observation model; RMSE grows by ~30% vs PF variants and log-likelihood trails UKF slightly.
- **UKF**: sigma points remain stable but still under-estimate variance once observation noise is tightened; ESS metrics are N/A (deterministic filter).
- **PF vs DPF**: DPF maintains ESS ratio above 0.55 across seeds whereas vanilla PF frequently dips below the 0.5 resampling threshold after step 10, highlighting the degeneracy issue documented in the gap plan.
- **Failure cases**: When observation noise std < 0.1 the EKF covariance collapses and residual spikes appear around step 10; UKF sigma points saturate near the highly nonlinear region (step 12), a behaviour captured in the ESS and RMSE traces.

## Notes & Caveats
- **Retracing warnings**: TensorFlow reported repeated tracing in `benchmark_flow()` when flows are invoked inside the time loop. Consider caching compiled `tf.function`s or enabling `reduce_retracing=True` to improve performance and suppress warnings.
- **Single-seed evaluation**: Baseline tables above stem from seed 0; see the multi-seed sections for aggregated confidence intervals.
- **CPU-only run**: GPU acceleration could substantially reduce runtimes, particularly for EDH/LEDH and OT resampling.

## Recommended Follow-Up
- Tighten differentiable PF hyperparameters (e.g., `mix_with_uniform`, Sinkhorn epsilon) to improve log-likelihood without sacrificing ESS.
- Profile flow routines to identify hotspots; experiment with analytical Jacobians or batched linear solves to reduce EDH/LEDH runtime.
- Generate plots (ESS trajectories, residuals, transport plan entropy) for inclusion in the Question&nbsp;2 deliverable.
- Evaluate bonus-scope extensions after addressing retracing and multi-seed benchmarking.


## New Additions (Automation Ready)
- `benchmark.py` now records flow log-Jacobian statistics and exposes PF-PF runs (`PF_PF_EDH`, `PF_PF_LEDH`, `PF_PF_Kernel*`). Re-run `python -m mlcoe_q2.pipelines.benchmark` to populate the new `pfpf_results` block in the JSON output.
- Flow diagnostics include per-step log-det means; use them to assess numerical stability before reproducing Li (2017) figures.
- Differentiable PF benchmarking benefits from the epsilon schedule and gradient-checked implementation—tune `--mix-with-uniform`, `--ot-epsilon`, and `--sinkhorn-tolerance` directly in the CLI.

## Multi-Seed Results (Filters)
Aggregated over 5 seeds from `reports/artifacts/benchmark_filters_multiseed.json`.

| Method | Runtime (s) | Peak Mem (KB) | Log-Likelihood | Mean ESS |
| --- | ---: | ---: | ---: | ---: |
| PF | 52.18 ± 1.94 | 33565.63 ± 326.51 | -30.33 ± 14.50 | 169.56 ± 10.21 |
| Differentiable PF | 51.68 ± 1.79 | 34056.45 ± 190.05 | -103.64 ± 15.79 | 181.47 ± 23.36 |
| EKF | 10.61 ± 1.66 | 2414.06 ± 89.65 | -10.09 ± 0.72 | — |
| UKF | 1.84 ± 0.10 | 2206.99 ± 14.51 | -9.99 ± 0.90 | — |

Notes:
- ESS only applies to PF variants; EKF/UKF do not maintain particles.
- See `reports/q2/status/filter_status.md` for a Markdown status snapshot generated by the automation script.

### Discussion
- **Likelihood vs ESS**: `EKF/UKF` achieve the best (least negative) log-likelihoods. `Differentiable PF` improves ESS vs `PF` but degrades log-likelihood, likely due to OT regularization strength or mismatch.
- **Runtime/Memory**: `UKF` is fastest; `EKF` is stable at ~10.6s. `PF/DPF` dominate runtime and memory due to resampling/transport; DPF is similar runtime to PF in this setup.
- **Next**: Tune DPF (`ot_epsilon`, `ot_num_iters`, `mix_with_uniform`) to recover likelihood while retaining ESS gains; consider GPU for larger particles.

## PF-PF Results (Multi-Seed)
Aggregated over 5 seeds from `reports/artifacts/pfpf_stochastic_multiseed.json`.

| Method | Runtime (s) | Peak Mem (KB) | Log-Likelihood | Mean ESS |
| --- | ---: | ---: | ---: | ---: |
| PF_PF_LEDH | 23613.95 ± 1597.64 | 146100.40 ± 520.79 | -17.81 ± 6.26 | 179.18 ± 5.78 |
| PF_PF_Stochastic | 15459.71 ± 672.27 | 239114.86 ± 87.99 | -8.37 ± 10.32 | 176.48 ± 8.91 |

Notes:
- Stochastic flow is ~35% faster with better (less negative) log-likelihood and similar ESS.
- Memory usage increases for Stochastic due to diffusion and additional computations.

### Discussion
- **Speed**: Stochastic flow reduces runtime by ~35% vs LEDH at this scale.
- **Accuracy/Stability**: Better average log-likelihood with comparable ESS suggests diffusion improves exploration without harming weight degeneracy.
- **Memory**: Higher peak memory for Stochastic reflects the additional noise integration and per-step ops; acceptable trade-off given speed/LL gains.

## Li (2017)-Style Diagnostics (Kernel Flows)
Figures generated via `mlcoe_q2/pipelines/plot_flow_diagnostics.py`.

Artifacts index: `reports/q2/status/li2017_plots.md`

Figures (by flow variant):
- KernelScalar: `reports/figures/li2017_KernelScalar_residuals.png`, `li2017_KernelScalar_movement.png`, `li2017_KernelScalar_logjac.png`
- KernelDiagonal: `reports/figures/li2017_KernelDiagonal_residuals.png`, `li2017_KernelDiagonal_movement.png`, `li2017_KernelDiagonal_logjac.png`
- KernelMatrix: `reports/figures/li2017_KernelMatrix_residuals.png`, `li2017_KernelMatrix_movement.png`, `li2017_KernelMatrix_logjac.png`

### KernelScalar
![KernelScalar residuals](figures/li2017_KernelScalar_residuals.png)
![KernelScalar movement](figures/li2017_KernelScalar_movement.png)
![KernelScalar |log-J|](figures/li2017_KernelScalar_logjac.png)

### KernelDiagonal
![KernelDiagonal residuals](figures/li2017_KernelDiagonal_residuals.png)
![KernelDiagonal movement](figures/li2017_KernelDiagonal_movement.png)
![KernelDiagonal |log-J|](figures/li2017_KernelDiagonal_logjac.png)

### KernelMatrix
![KernelMatrix residuals](figures/li2017_KernelMatrix_residuals.png)
![KernelMatrix movement](figures/li2017_KernelMatrix_movement.png)
![KernelMatrix |log-J|](figures/li2017_KernelMatrix_logjac.png)

## High-Dimensional Kernel Flows (Hu, 2021)
Demonstration of observed-marginal collapse prevention using matrix-valued kernels in higher dimensions.

Artifacts index: `reports/q2/status/hu2021_plots.md`

### KernelScalar (state=16, obs=4)
![Observed variance](figures/hu2021_KernelScalar_obsvar.png)

### KernelDiagonal (state=16, obs=4)
![Observed variance](figures/hu2021_KernelDiagonal_obsvar.png)

### KernelMatrix (state=16, obs=4)
![Observed variance](figures/hu2021_KernelMatrix_obsvar.png)

## Li (2017) PF-PF Reproduction (EDH vs LEDH)
Seeded PF-PF runs to visualize ESS, |log-J|, and normalized per-step log-likelihood.

Artifacts index: `reports/q2/status/li2017_pfpf_reproduction.md`

### EDH
![EDH ESS](figures/li2017_pfpf_EDH_ess.png)
![EDH |log-J|](figures/li2017_pfpf_EDH_logj.png)
![EDH per-step LL (normalized)](figures/li2017_pfpf_EDH_loglik.png)

### LEDH
![LEDH ESS](figures/li2017_pfpf_LEDH_ess.png)
![LEDH |log-J|](figures/li2017_pfpf_LEDH_logj.png)
![LEDH per-step LL (normalized)](figures/li2017_pfpf_LEDH_loglik.png)

## Differentiable PF: Resampling Comparisons (Multi-Seed)
Compare soft weights (no transport) vs OT low-iter vs full OT across seeds.

Artifacts index: `reports/q2/status/dpf_comparisons.md`

- **Runtime/Memory**: OT is slightly faster on average at this scale; memory broadly similar.
- **Likelihood vs ESS**: Soft yields best (least negative) log-likelihood but lowest ESS. OT increases ESS substantially with a trade-off in likelihood.

## PF-PF (Dai 2022-style) Parameter Sweep (Multi-Seed)
LEDH vs stochastic flow configurations approximating Dai (2022) (smaller step-size, more steps, varying diffusion).

Artifacts index: `reports/q2/status/pfpf_dai22.md`

- **Accuracy**: `SPF_B (step=0.6, steps=8, diff=0.10)` improves log-likelihood over LEDH with similar ESS.
- **Runtime/Memory**: Some SPF settings are faster than LEDH; higher diffusion/steps increase memory.

### New multi-seed artifacts (Dai'22 variants)
- `reports/artifacts/pfpf_dai22_seed_{0..4}.json`

Across five seeds, the stochastic flow variant `SPF_B`:
- **Log-likelihood**: improves over `LEDH` in most seeds (4/5), with one seed where `LEDH` is higher.
- **Runtime**: comparable or slightly lower on average vs `LEDH` at this scale (tens of thousands of seconds on CPU-only runs).
- **ESS**: remains in a similar range to `LEDH`.
- **Memory**: ~2× higher peak memory than `LEDH`, consistent with additional diffusion computations.

These results are consistent with the earlier PF-PF multi-seed comparison where stochastic flows offered a better likelihood–runtime trade-off at the cost of memory.


## Bonus: PMMH vs HMC with Differentiable PF
This bonus experiment compares a PMMH baseline (using a standard PF likelihood estimate) against an HMC sampler that leverages a differentiable particle filter (DPF) to obtain gradients on a small nonlinear SSM.

- Script: `mlcoe_q2/pipelines/pmmh_vs_hmc_dpf.py`
- Example output: `reports/artifacts/pmmh_vs_hmc.json`
- Status snapshot: `reports/q2/status/bonus_pmmh_vs_hmc.md`

Run:

```bash
PYTHONPATH=. python -m mlcoe_q2.pipelines.pmmh_vs_hmc_dpf \
  --config configs/q2/pmmh_vs_hmc_dpf.json
```

Notes:
- CPU-friendly scaffold to validate feasibility and compare acceptance/ESS trends.
- Prior on `phi` is `N(0,1)`; `phi` parameterizes the observation noise log-std.
- Enable `--eager` (included in the config) to bypass long graph-compilation overhead; even so expect a few minutes per run because the HMC kernel differentiates through the DPF log-likelihood.
- With the reference config, PMMH acceptance hovers near 25% while HMC lands around 70% with tighter posterior contraction (see artifact JSON for chains).

## Bonus: Neural OT Acceleration

- Script: `mlcoe_q2/pipelines/neural_ot_acceleration`
- Artifact: `reports/artifacts/neural_ot_acceleration.json`
- Status snapshot: `reports/q2/status/bonus_neural_ot.md`

Key observations:
- Trained dense network approximates Sinkhorn transport for 8-particle batches with <1e-3 row/column mass drift on held-out data.
- Averaged over 200 validation batches, neural inference achieves ~4× faster per-sample throughput than classical Sinkhorn on CPU.
- Saved accelerator integrates with `differentiable_particle_filter(..., resampling_method="neural_ot")` for gradient-based workflows.

## Bonus: Neural State-Space Inference

- Script: `mlcoe_q2.pipelines.neural_state_space_inference`
- Artifact: `reports/artifacts/neural_state_space_inference.json`
- Status snapshot: `reports/q2/status/bonus_neural_ssm.md`

Key observations:
- LSTM-style latent dynamics provide a nonlinear benchmark for gradient vs particle-MCMC inference.
- With the compact four-step configuration, DPF-HMC reaches near-unity acceptance (~1.00) and ESS ≈5–6 while Particle Gibbs sits around 0.71 acceptance and lower ESS despite a shorter runtime.
- Particle Gibbs remains a viable baseline but mixes more slowly, highlighting the benefit of differentiable (or neural-OT accelerated) resampling for gradient-informed samplers.
