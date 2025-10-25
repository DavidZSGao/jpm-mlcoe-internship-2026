# Question 2 — Particle Flow Filtering & Differentiable PF

This package contains state-space models, particle-filter variants, particle-flow implementations, and experiment pipelines for the Question 2 deliverables.

## Layout
- `data/` — synthetic state-space model definitions (linear-Gaussian and nonlinear benchmarks).
- `models/filters/` — Kalman-family filters, sequential importance resampling (PF), differentiable PF, and particle-flow particle filters.
- `models/flows/` — deterministic (EDH/LEDH/Kernel) and stochastic particle flows.
- `evaluation/` — helpers to aggregate multi-seed benchmark metrics and render Markdown status reports.
- `pipelines/` — runnable experiment and reporting CLIs (multi-seed sweeps, diagnostics, PF-PF reproductions, PMMH vs HMC, etc.).
- `utils/` — shared CLI helpers (JSON/YAML configs, filesystem utilities).

Tests for this module live under `tests/` (see `pytest -k q2`).

## Usage
Each CLI accepts inline arguments and can load defaults from JSON/YAML via `--config` (presets live under `configs/q2`). Examples:

- Filters vs PF benchmarks (multi-seed):
  ```bash
  python -m mlcoe_q2.pipelines.run_multiseed_benchmarks --config configs/q2/run_multiseed_benchmarks.json
  ```
- PF-PF stochastic vs LEDH sweep:
  ```bash
  python -m mlcoe_q2.pipelines.run_pfpf_stochastic_multiseed --config configs/q2/run_pfpf_stochastic_multiseed.json
  ```
- Differentiable PF resampling comparison (soft vs OT):
  ```bash
  python -m mlcoe_q2.pipelines.run_dpf_comparisons_multiseed --config configs/q2/run_dpf_comparisons_multiseed.json
  ```
- Dai (2022) stochastic flow configurations:
  ```bash
  python -m mlcoe_q2.pipelines.run_pfpf_dai22_multiseed --config configs/q2/run_pfpf_dai22_multiseed.json
  ```
- Nonlinear filter diagnostics with plots:
  ```bash
  python -m mlcoe_q2.pipelines.nonlinear_filter_diagnostics --config configs/q2/nonlinear_filter_diagnostics.json
  ```
- LGSSM Kalman validation report:
  ```bash
  python -m mlcoe_q2.pipelines.lgssm_validation --config configs/q2/lgssm_validation.json
  ```
- Li (2017) kernel-flow diagnostics and Hu (2021) high-dimensional plots:
  ```bash
  python -m mlcoe_q2.pipelines.plot_flow_diagnostics --config configs/q2/plot_flow_diagnostics.json
  python -m mlcoe_q2.pipelines.plot_highdim_kernel_flows --config configs/q2/plot_highdim_kernel_flows.json
  ```
- PMMH vs HMC with differentiable PF gradients:
  ```bash
  python -m mlcoe_q2.pipelines.pmmh_vs_hmc_dpf --config configs/q2/pmmh_vs_hmc_dpf.json
  ```
- Neural OT resampling accelerator training:
  ```bash
  python -m mlcoe_q2.pipelines.neural_ot_acceleration --config configs/q2/neural_ot_acceleration.json
  ```
- Neural state-space inference (DPF-HMC vs Particle Gibbs):
  ```bash
  python -m mlcoe_q2.pipelines.neural_state_space_inference --config configs/q2/neural_state_space_inference.json
  ```

Generated artifacts mirror the reporting layout under `reports/q2/` (status Markdown + figures) and `reports/artifacts/` (JSON summaries).

### Quickstart presets

The defaults above regenerate the full submission artifacts and can take multiple
hours on CPU-only machines. For faster smoke tests, replace the config paths with
the lightweight variants under `configs/q2/quickstart/`. They emit outputs into
`reports/artifacts/quickstart/` and are summarised in `reports/q2/quickstart.md`.

## Notes
- All TensorFlow routines set explicit seeds before simulation to keep multi-seed comparisons reproducible.
- Multi-seed pipelines call into `mlcoe_q2.evaluation.aggregate_metrics` to compute means/std-devs and auto-render Markdown tables consistent with the Question 1 automation style.
- Config presets can be customised per run; explicit CLI flags always override config defaults.
