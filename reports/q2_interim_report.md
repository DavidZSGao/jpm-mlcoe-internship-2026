# Question 2 — Interim Report

Prepared for the October checkpoint. This report summarizes the current status of filtering and particle-flow experiments, key results, and the gap-closure plan to finalize Part 2.

## 1. Executive Summary
- Implemented EKF, UKF, standard PF, Differentiable PF (OT resampling), and multiple particle flows (EDH, LEDH, Kernel, Stochastic) under `mlcoe_q2/`.
- Baseline single-seed benchmarks are complete and documented in `reports/q2_benchmark_report.md` with artifacts under `reports/artifacts/`.
- Next step in progress: multi-seed benchmarking for robustness, and PF-PF with stochastic flow proposal.

## 2. Methods Implemented
- **Classical filters**: `extended_kalman_filter()`, `unscented_kalman_filter()` in `mlcoe_q2/filters/`.
- **Sequential Monte Carlo**: `particle_filter()` with standard resampling and diagnostics.
- **Differentiable PF**: `differentiable_particle_filter()` with entropy-regularized OT resampling (Sinkhorn), hyperparameters exposed (`mix_with_uniform`, `ot_epsilon`, `ot_num_iters`).
- **Particle flows**: `ExactDaumHuangFlow`, `LocalExactDaumHuangFlow`, `KernelEmbeddedFlow` (scalar/diagonal/matrix), `StochasticParticleFlow` in `mlcoe_q2/flows/`.
- **PF-PF**: `particle_flow_particle_filter()` with flow proposals (EDH/LEDH/Kernel; stochastic flow proposal planned).

## 3. Experimental Setup
- Nonlinear SSM defined in `mlcoe_q2/experiments/benchmark.py::_build_nonlinear_model()`.
- Default: 15 timesteps, 256 particles for PF/DPF; CPU run in TensorFlow.
- Benchmark driver: `mlcoe_q2/experiments/benchmark.py::run_benchmark_suite()`.

## 4. Results (Single Seed)
See `reports/q2_benchmark_report.md` for the full table. Highlights:
- **EKF/UKF** achieved higher (less negative) log-likelihoods and lowest runtime.
- **Differentiable PF** increased ESS vs PF but reduced log-likelihood (potential mismatch or hyperparameter tuning needed).
- **Kernel flow** reduced residuals most effectively at moderate runtime; EDH/LEDH are CPU-heavy.
- Artifacts and plots:
  - `reports/artifacts/*.json` (LGSSM, nonlinear diagnostics, latest benchmark)
  - `reports/figures/lgssm_validation_means.png`, `nonlinear_rmse.png`, `nonlinear_ess.png`

## 5. Multi-Seed Benchmarking (Completed)
Aggregated over 5 seeds from `reports/artifacts/benchmark_filters_multiseed.json`.

| Method | Runtime (s) | Peak Mem (KB) | Log-Likelihood | Mean ESS |
| --- | ---: | ---: | ---: | ---: |
| PF | 52.18 ± 1.94 | 33565.63 ± 326.51 | -30.33 ± 14.50 | 169.56 ± 10.21 |
| Differentiable PF | 51.68 ± 1.79 | 34056.45 ± 190.05 | -103.64 ± 15.79 | 181.47 ± 23.36 |
| EKF | 10.61 ± 1.66 | 2414.06 ± 89.65 | -10.09 ± 0.72 | — |
| UKF | 1.84 ± 0.10 | 2206.99 ± 14.51 | -9.99 ± 0.90 | — |

See `reports/q2/status/filter_status.md` for the generated status snapshot.

## 6. PF-PF with Stochastic Flow (Planned)
- Compare PF-PF using `StochasticParticleFlow` vs LEDH as proposals.
- Metrics: log-likelihood, ESS, runtime/memory; stability via flow log-Jacobians.
- Artifact: `reports/artifacts/pfpf_stochastic_multiseed.json`.

## 7. Reproducing Li (2017) Plots (Planned)
- Use `benchmark.py` flow diagnostics to recreate comparative plots.
- Deliverables: flow movement, residual before/after, log-det Jacobian trends.

## 8. Notes & Optimizations
- Address TensorFlow retracing: cache `tf.function`s or use `reduce_retracing=True` in hot paths.
- Consider GPU for EDH/LEDH and OT resampling to reduce runtime.
- Tune DPF hyperparameters (Sinkhorn epsilon, iterations, mix-with-uniform) to improve likelihood while keeping ESS gains.

## 9. Conclusion
- Core implementations are complete and validated on single seed.
- Multi-seed filters are running; flows and PF-PF comparisons to follow.
- Final deliverable will include robustness tables, Li (2017) reproduction plots, and analysis narrative.
