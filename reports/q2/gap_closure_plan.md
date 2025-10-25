# Question 2 Gap Closure Plan

## Part 1A — Kalman Filter for LGSSM
- ✅ Validation CLI `mlcoe_q2/pipelines/lgssm_validation.py` now exports JSON + plots (see `reports/artifacts/lgssm_validation_summary.json` and `reports/figures/lgssm_validation_means.png`).
- ✅ Conditioning metrics and Joseph-form comparison captured in the CLI summary payload.
- ✅ Figures and stability commentary embedded in `reports/q2/benchmark_report.md` and summarised in the interim report.

## Part 1B — Nonlinear SSM with EKF/UKF/PF
- ✅ Diagnostics runner (`mlcoe_q2/pipelines/nonlinear_filter_diagnostics.py`) produces RMSE/NLL/ESS artifacts and figures; includes per-seed aggregation and Sinkhorn tuning hooks.
- ✅ Multi-seed RMSE/ESS figures integrated into `reports/q2/benchmark_report.md` and referenced in the interim narrative.
- ✅ Documented EKF/UKF linearisation limits and sigma-point failures in the interim report using the diagnostics artifacts.

## Part 1C — Deterministic & Kernel Flows
- ✅ EDH/LEDH now accumulate log-Jacobians via automatic differentiation; diagnostics log mean log-det increments.
- ✅ Kernel flow exposes scalar/diagonal/matrix metrics with per-step conditioning reports.
- ✅ PF-PF benchmarks (EDH/LEDH/Kernel) integrated via `particle_flow_particle_filter` and surfaced in `mlcoe_q2/pipelines/benchmark.py`.
- ✅ Reproduced Li (2017) flow diagnostics (status pages + figures in `reports/q2/status/li2017_plots.md`).

## Part 2 — Stochastic Particle Flow & Differentiable PF
- ✅ Stochastic flow reuses Hessian-based log-Jacobian tracking and plugs into PF-PF benchmarking.
- ✅ Differentiable PF supports epsilon schedules, Sinkhorn tolerances, and passes gradient vs finite-difference tests.
- ✅ Multi-seed Dai (2022) stochastic flow sweep compared against LEDH (`reports/q2/status/pfpf_dai22.md`).
- ✅ PMMH vs HMC comparison captured via `mlcoe_q2/pipelines/pmmh_vs_hmc_dpf.py` with summary in `reports/q2/status/bonus_pmmh_vs_hmc.md` and artifact JSON.

## Bonus Scope (Completed)
- ✅ Trained neural OT accelerator and benchmarked runtime vs Sinkhorn (`mlcoe_q2.pipelines.neural_ot_acceleration`).
- ✅ Delivered neural state-space inference comparison (DPF-HMC vs Particle Gibbs) with reproducible CLI and artifact.
