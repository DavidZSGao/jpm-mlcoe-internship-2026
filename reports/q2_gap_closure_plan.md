# Question 2 Gap Closure Plan

## Part 1A — Kalman Filter for LGSSM
- ✅ Validation CLI `mlcoe_q2/experiments/lgssm_validation.py` now exports JSON + plots (see `reports/artifacts/lgssm_validation_summary.json` and `reports/figures/lgssm_validation_means.png`).
- ✅ Conditioning metrics and Joseph-form comparison captured in the CLI summary payload.
- ☐ Incorporate figures/tables into the final write-up narrative.

## Part 1B — Nonlinear SSM with EKF/UKF/PF
- ✅ Diagnostics runner (`mlcoe_q2/experiments/nonlinear_filter_diagnostics.py`) produces RMSE/NLL/ESS artifacts and figures; includes per-seed aggregation and Sinkhorn tuning hooks.
- ☐ Generate multi-seed plots for the report (ESS ratio trajectories, RMSE bands).
- ☐ Document EKF/UKF failure cases using the new outputs.

## Part 1C — Deterministic & Kernel Flows
- ✅ EDH/LEDH now accumulate log-Jacobians via automatic differentiation; diagnostics log mean log-det increments.
- ✅ Kernel flow exposes scalar/diagonal/matrix metrics with per-step conditioning reports.
- ✅ PF-PF benchmarks (EDH/LEDH/Kernel) integrated via `particle_flow_particle_filter` and surfaced in `benchmark.py`.
- ☐ Reproduce Li (2017) plots using the upgraded benchmarking pipeline.

## Part 2 — Stochastic Particle Flow & Differentiable PF
- ✅ Stochastic flow reuses Hessian-based log-Jacobian tracking and plugs into PF-PF benchmarking.
- ✅ Differentiable PF supports epsilon schedules, Sinkhorn tolerances, and passes gradient vs finite-difference tests.
- ☐ Extend benchmarks to compare optimized stochastic flow proposals against LEDH across multiple seeds.
- ☐ Investigate HMC/PMMH coupling atop the differentiable PF outputs.

## Bonus Scope (Pending Prioritization)
- ☐ Prioritize Bonus tracks after core benchmarking plots land.
- ☐ Outline resource requirements for neural OT acceleration and DPF-HMC extensions.
