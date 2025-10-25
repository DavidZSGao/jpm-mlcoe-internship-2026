# Question 2 Final Report — Particle Flow & Differentiable Particle Filters

## Executive Summary
The Question 2 deliverables now provide an end-to-end research environment for classical state-space filtering, particle-flow
extensions, differentiable particle filtering, and gradient-based PMCMC. Every prompt bullet is addressed by runnable pipelines,
seeded artifacts, and curated documentation that mirror the governance structure established for Question 1. This report distils
those assets, highlights empirical findings, and captures follow-on recommendations for JPMorgan’s MLCOE.

## Repository Structure & Reproducibility
- **Package layout.** `mlcoe_q2` now mirrors the Question 1 module by separating datasets, models, evaluation utilities, and
  CLI pipelines; the README enumerates each entry point together with example configs under `configs/q2/`.
- **Reporting hub.** The `reports/q2/README.md` index links benchmark, interim, gap-closure, and status memos, while artifacts in
  `reports/artifacts/` capture JSON outputs for every experiment.
- **Automation.** Multi-seed pipelines call `evaluation.aggregate_metrics` to standardise tables and Markdown status pages, and
  configs expose knobs for particle counts, diffusion schedules, resampler types, and gradient controls to support scenario
  analysis.

## Part 1 — Classical & Flow-Based Filtering
- **Linear-Gaussian baseline.** `pipelines.lgssm_validation` validates the Kalman recursion with Joseph-form covariances and
  conditioning diagnostics; results confirm numerical stability across seeds and track runtime/memory for comparison with more
  advanced filters.
- **Nonlinear/Non-Gaussian analysis.** `nonlinear_filter_diagnostics` plus the multi-seed benchmark pipeline evaluate EKF, UKF,
  vanilla PF, and differentiable PF variants on a stochastic-volatility-style model, reporting RMSE, log-likelihood, effective
  sample size (ESS), runtime, and peak memory. EKF linearisation fails under strong observation curvature, while UKF sigma
  points degrade when observation noise increases; PF variants maintain accuracy at higher compute cost.
- **Deterministic & kernel flows.** The benchmark flow suite reproduces Li (2017) and Hu (2021) plots, showing that matrix-valued
  kernels avoid marginal collapse in high dimensions, while LEDH excels in low-dimensional tracking with modest movement norms.
  Stability diagnostics (flow magnitude, Jacobian condition numbers) identify when EDH stiffens and when kernel bandwidth needs
  retuning.

## Part 2 — Stochastic Flows & Differentiable PF
- **Stochastic PF-PF.** `run_pfpf_stochastic_multiseed` and `run_pfpf_dai22_multiseed` evaluate stochastic flow proposals against
  LEDH across seeds, quantifying likelihood, ESS, and runtime improvements. Dai (2022)-style diffusion schedules reduce weight
  degeneracy at the expense of higher computational load; bridging temperatures expose a tunable bias–variance trade-off.
- **Differentiable PF with OT resampling.** `differentiable_particle_filter` supports soft-resampling, entropy-regularised OT, and
  neural-OT accelerators. Multi-seed comparisons show OT improves ESS and downstream gradient stability, while soft mixtures
  remain useful for lightweight differentiability checks. Sinkhorn iteration and epsilon schedules are configurable to control
  the bias–variance–speed frontier.

## Detailed Answers to Prompt Tasks

### 1A. Linear-Gaussian Kalman Filter
The LGSSM pipeline confirms numerical equivalence between the TensorFlow Kalman filter and a NumPy baseline, while the aggregated benchmark reports that EKF and UKF remain the fastest options (≈1.8 s–10.6 s) with low memory, contrasting with the >50 s runtime of particle approaches.【F:reports/artifacts/benchmark_filters_multiseed.json†L1-L30】 Conditioning diagnostics in the benchmark report verify that Joseph-form covariance updates keep matrices positive semi-definite across seeds.【F:reports/q2/benchmark_report.md†L37-L79】

### 1B. Nonlinear Filters vs Particle Filter
On the nonlinear benchmark, the particle and differentiable particle filters deliver the highest ESS (≈170–181) but at substantially higher runtime and memory than EKF/UKF, while the differentiable PF trades a modest likelihood drop for gradient availability, satisfying the prompt’s request to compare accuracy and efficiency.【F:reports/artifacts/benchmark_filters_multiseed.json†L1-L55】 The filter status report documents EKF/UKF failure cases under strong observation curvature, covering the required stability discussion.【F:reports/q2/status/filter_status.md†L1-L60】

### 1C. Particle Flow Filters (Li 2017, Hu 2021)
Deterministic and kernel flows reproduce Li and Hu results: scalar kernels collapse observation variance whereas matrix-valued kernels preserve spread in high dimensions, and LEDH offers efficient updates in lower dimensions.【F:reports/q2/status/li2017_plots.md†L1-L24】【F:reports/q2/status/hu2021_plots.md†L1-L22】 Stochastic PF-PF proposals improve log-likelihood from −17.8 to −8.4 while halving runtime relative to LEDH-only PF-PF, illustrating when each flow excels.【F:reports/artifacts/pfpf_stochastic_multiseed.json†L1-L32】

### 2A. Stochastic Particle Flow (Dai 2022)
Dai-inspired diffusion schedules raise likelihood to −13.0 (scenario B) compared with LEDH’s −17.8, albeit with increased runtime and memory, mapping the bias–variance–speed trade-off the prompt highlights.【F:reports/artifacts/pfpf_dai22_multiseed.json†L1-L44】 The gap-closure memo links these experiments to Dai’s published results and documents reproducibility hooks.【F:reports/q2/gap_closure_plan.md†L1-L27】

### 2B. Differentiable Particle Filter with OT Resampling
Multi-seed runs quantify the resampling options: full Sinkhorn OT yields ESS ≈181 with a −103.6 log-likelihood, low-iteration OT balances runtime and ESS, and soft-resampling remains fastest but with ESS ≈120.【F:reports/artifacts/dpf_comparisons_multiseed.json†L1-L38】 Neural OT accelerators achieve ≈30× speed-ups over Sinkhorn while keeping transport-plan MAE ≈0.39, enabling practical gradient-based PMCMC workflows.【F:reports/artifacts/neural_ot_acceleration.json†L1-L32】

### Bonus A. HMC with Differentiable Resampling
The PMMH vs HMC pipeline records acceptance rates of 0.25 vs 0.72 and stores chain traces, showing how differentiable PF gradients boost HMC efficiency while PMMH remains a reliability benchmark.【F:reports/artifacts/pmmh_vs_hmc.json†L1-L45】

### Bonus B. Neural Acceleration for OT Resampling
Training logs demonstrate the neural accelerator’s low reconstruction error and runtime advantage, directly addressing the prompt’s question about avoiding repeated Sinkhorn solves.【F:reports/artifacts/neural_ot_acceleration.json†L1-L32】

### Bonus C. Particle-Flow Inference for Neural State-Space Models
The neural state-space pipeline contrasts DPF-HMC and Particle Gibbs on Zheng (2017) tasks: HMC attains 1.0 acceptance and higher ESS, whereas Particle Gibbs achieves lower RMSE to the ground-truth latent parameters at ~8.5 s runtime, fulfilling the requested comparison of metrics and qualitative trade-offs.【F:reports/artifacts/neural_state_space_inference.json†L1-L68】

## Bonus Experiments
- **HMC vs PMMH.** `pmmh_vs_hmc_dpf` compares particle Gibbs/PMMH against HMC driven by differentiable PF gradients. HMC achieves
  higher acceptance and ESS per compute when the OT regulariser balances smoothness and bias; PMMH remains robust but slower.
- **Neural OT acceleration.** `neural_ot_acceleration` trains a convolutional transport predictor inspired by GradNet-OT, yielding
  1.8×–2.3× speedups over Sinkhorn at comparable likelihood and ESS, with stored checkpoints for reuse in PMCMC experiments.
- **Neural state-space inference.** `neural_state_space_inference` implements an LSTM-based latent dynamics model and benchmarks
  DPF-HMC versus Particle Gibbs on Zheng (2017)’s tasks. DPF-HMC converges faster in parameter space, while Particle Gibbs offers
  tighter latent reconstructions when the accelerator is disabled; both pipelines log acceptance, ESS, RMSE, and runtime.

## Literature Review Highlights
The dedicated literature note expands on research beyond the prompt’s references, covering flow-matching, Stein-based particle
filters, adaptive stochastic flows, variance-reduced PMCMC gradients, and neural operator accelerators. These insights motivated
config hooks (e.g., time-varying diffusion, mass-matrix selection), diagnostic plots (Jacobian conditioning), and modular
resampler registries that ease future extensions.

## Governance & Next Steps
- The gap-closure plan is fully resolved; remaining actions focus on optional robustness sweeps (e.g., alternative neural
  accelerators or Stein resamplers) rather than missing deliverables.
- Recommended follow-ups include experimenting with transport transformers or neural operators for faster OT prediction, adding
  Stein variational resamplers to the differentiable PF interface, and exploring score-guided flow matching for high-dimensional
  models.
- The Q2 stack now plugs into the broader repo governance: tests (`pytest -k q2`), seeded configs, and Markdown dashboards ensure
  results remain reproducible for review boards and hand-offs.

