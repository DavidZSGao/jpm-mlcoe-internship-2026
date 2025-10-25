# Question 2 Response Summary

This memo documents how the Question 2 codebase satisfies the state-space, particle-flow, and differentiable particle-filter tasks, mapping each prompt component to runnable pipelines, artifacts, and reports.

## Platform Overview
- `mlcoe_q2` packages the synthetic datasets, filter/flow implementations, evaluation helpers, and CLI pipelines, mirroring the reproducibility tooling built for Question 1; the README lists every runnable entry point (Kalman validation, nonlinear diagnostics, PF-PF sweeps, differentiable PF benchmarks, PMMH vs HMC, neural accelerators, and neural SSM inference).【F:mlcoe_q2/README.md†L1-L62】
- The deliverables index under `reports/q2/` inventories the narrative reports, status snapshots, and artifacts so each experiment can be traced from command to Markdown summary to JSON outputs.【F:reports/q2/README.md†L1-L58】

## Part 1 — Classical & Flow-Based Filtering
### 1A. Linear-Gaussian SSM with Kalman Filter
- `mlcoe_q2.pipelines.lgssm_validation` simulates a conditioned LGSSM, runs the TensorFlow Kalman filter, recomputes the recursion in NumPy double precision, and exports Joseph-form covariance checks plus summary JSON/plots that feed the benchmark report.【F:mlcoe_q2/pipelines/lgssm_validation.py†L2-L198】
- The benchmark report captures the Kalman validation metrics and conditioning analysis alongside runtime/memory comparisons for all filters.【F:reports/q2/benchmark_report.md†L1-L55】

### 1B. Nonlinear/Non-Gaussian SSM with EKF, UKF, and PF
- `mlcoe_q2.pipelines.nonlinear_filter_diagnostics` builds a nonlinear benchmark model, measures RMSE/log-likelihood/ESS per filter, and logs plots highlighting EKF/UKF failure modes, while `run_multiseed_benchmarks` sweeps seeds to aggregate PF, DPF, EKF, and UKF metrics into artifacts and Markdown tables.【F:mlcoe_q2/pipelines/nonlinear_filter_diagnostics.py†L2-L200】【F:mlcoe_q2/pipelines/run_multiseed_benchmarks.py†L1-L114】
- Multi-seed summaries and diagnostics are surfaced in the benchmark report’s filter sections, documenting likelihood, ESS, runtime, and memory trade-offs together with guidance on tuning OT regularisation.【F:reports/q2/benchmark_report.md†L37-L105】

### 1C. Deterministic and Kernel Particle Flows
- The benchmark suite constructs EDH, LEDH, kernel, and stochastic flows, profiles PF-PF variants, and records flow diagnostics (movement, residuals, log-Jacobians) for downstream reporting and Li/Hu plot reproduction.【F:mlcoe_q2/pipelines/benchmark.py†L1-L200】【F:mlcoe_q2/pipelines/benchmark.py†L200-L399】
- Li (2017) and Hu (2021) style plots plus PF-PF comparisons are published in the benchmark report and linked status pages, demonstrating when kernel matrices prevent marginal collapse and how stochastic flows trade runtime for memory.【F:reports/q2/benchmark_report.md†L81-L177】

## Part 2 — Stochastic Particle Flow & Differentiable PF
### Stochastic Particle Flow for PF-PF
- `run_pfpf_stochastic_multiseed` contrasts LEDH and stochastic proposals across seeds, while `run_pfpf_dai22_multiseed` extends the sweep with Dai (2022)-style diffusion schedules; both emit per-seed JSON, aggregate stats, and Markdown tables for the reporting hub.【F:mlcoe_q2/pipelines/run_pfpf_stochastic_multiseed.py†L1-L135】【F:mlcoe_q2/pipelines/run_pfpf_dai22_multiseed.py†L1-L112】
- The benchmark report summarises multi-seed outcomes, highlighting the runtime/log-likelihood gains from stochastic flows and the diffusion trade-offs explored in the Dai (2022) sweep.【F:reports/q2/benchmark_report.md†L81-L177】

### Differentiable PF with OT Resampling
- `mlcoe_q2.models.filters.differentiable_particle_filter` implements soft, Sinkhorn-OT (full/low-iter), and neural-OT resampling with ESS diagnostics, epsilon schedules, and transport-plan capture for gradient-based inference.【F:mlcoe_q2/models/filters/differentiable_pf.py†L1-L200】【F:mlcoe_q2/models/filters/differentiable_pf.py†L200-L330】
- `run_dpf_comparisons_multiseed` benchmarks soft vs OT-low vs full OT resampling, recording runtime, memory, likelihood, and ESS across seeds to populate the DPF status snapshot and benchmark report discussion.【F:mlcoe_q2/pipelines/run_dpf_comparisons_multiseed.py†L1-L148】【F:reports/q2/benchmark_report.md†L152-L167】

## Bonus Experiments
### HMC vs PMMH with Differentiable PF
- `mlcoe_q2.pipelines.pmmh_vs_hmc_dpf` builds a scalar-parameter nonlinear SSM, contrasts PMMH (standard PF likelihood) with HMC (DPF gradients), and writes acceptance/ESS/runtime diagnostics plus artifacts referenced in the report.【F:mlcoe_q2/pipelines/pmmh_vs_hmc_dpf.py†L1-L198】【F:reports/q2/benchmark_report.md†L180-L199】

### Neural OT Acceleration
- `mlcoe_q2.models.resampling.neural_ot` generates Sinkhorn training data, defines the accelerator network, and wraps saved models for inference-time transport prediction with optional reference-error checks.【F:mlcoe_q2/models/resampling/neural_ot.py†L1-L200】【F:mlcoe_q2/models/resampling/neural_ot.py†L200-L320】
- `mlcoe_q2.pipelines.neural_ot_acceleration` trains the accelerator, measures speedup vs Sinkhorn, and publishes JSON + Markdown summaries that feed the bonus status page and artifacts.【F:mlcoe_q2/pipelines/neural_ot_acceleration.py†L1-L118】

### Neural State-Space Inference
- `mlcoe_q2.pipelines.neural_state_space_inference` builds an LSTM-based SSM, runs DPF-HMC and Particle Gibbs (optionally with the neural OT accelerator), summarises acceptance/ESS/RMSE, and exports artifacts/status Markdown.【F:mlcoe_q2/pipelines/neural_state_space_inference.py†L1-L198】

## Documentation & Status Tracking
- The interim report narrates implementation coverage, diagnostic findings, and remaining polish, while the gap-closure plan now maps each prompt bullet to finished pipelines and status assets.【F:reports/q2/interim_report.md†L1-L142】【F:reports/q2/gap_closure_plan.md†L1-L27】
- Status dashboards under `reports/q2/status/` pair every experiment with reproducibility commands, and the consolidated benchmark report provides tables, plots, and reproduction steps for all Part 1, Part 2, and bonus deliverables.【F:reports/q2/README.md†L5-L37】【F:reports/q2/benchmark_report.md†L1-L209】
