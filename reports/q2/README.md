# Q2 Deliverables Index

## Reports
- `reports/q2/final_report.md`
- `reports/q2/q2_response_summary.md`
- `reports/q2/benchmark_report.md`
- `reports/q2/interim_report.md`
- `reports/q2/gap_closure_plan.md`
- Quick iterations: `reports/q2/quickstart.md`

## Status Snapshots
- Filters (multi-seed): `reports/q2/status/filter_status.md`
- PF-PF (Stochastic vs LEDH): `reports/q2/status/pfpf_status.md`
- Li (2017)-style Kernel flow plots: `reports/q2/status/li2017_plots.md`
- Hu (2021)-style high-dimensional Kernel plots: `reports/q2/status/hu2021_plots.md`
- Li (2017) PF-PF reproduction (EDH vs LEDH): `reports/q2/status/li2017_pfpf_reproduction.md`
- DPF resampling comparisons: `reports/q2/status/dpf_comparisons.md`
- PF-PF Dai (2022) sweep: `reports/q2/status/pfpf_dai22.md`
- PMMH vs HMC bonus comparison: `reports/q2/status/bonus_pmmh_vs_hmc.md`
- Neural OT acceleration: `reports/q2/status/bonus_neural_ot.md`
- Neural state-space inference (DPF-HMC vs PG): `reports/q2/status/bonus_neural_ssm.md`

## Key Artifacts
- Filters aggregate: `reports/artifacts/benchmark_filters_multiseed.json`
- Filters per-seed: `reports/artifacts/benchmark_filters_seed_{0..4}.json`
- PF-PF aggregate: `reports/artifacts/pfpf_stochastic_multiseed.json`
- PF-PF per-seed: `reports/artifacts/pfpf_stochastic_seed_{0..4}.json`
- Latest single-run snapshot: `reports/artifacts/benchmark_latest.json`
- Diagnostics: `reports/artifacts/nonlinear_diagnostics.json`
- LGSSM validation summary: `reports/artifacts/lgssm_validation_summary.json`
- PMMH vs HMC bonus run: `reports/artifacts/pmmh_vs_hmc.json`
- Neural OT accelerator metrics: `reports/artifacts/neural_ot_acceleration.json`
- Neural state-space inference comparison: `reports/artifacts/neural_state_space_inference.json`

## Figures
- Kernel flow diagnostics (Li 2017-style):
  - `reports/figures/li2017_KernelScalar_residuals.png`, `li2017_KernelScalar_movement.png`, `li2017_KernelScalar_logjac.png`
  - `reports/figures/li2017_KernelDiagonal_residuals.png`, `li2017_KernelDiagonal_movement.png`, `li2017_KernelDiagonal_logjac.png`
  - `reports/figures/li2017_KernelMatrix_residuals.png`, `li2017_KernelMatrix_movement.png`, `li2017_KernelMatrix_logjac.png`
- Additional: `reports/figures/nonlinear_rmse.png`, `reports/figures/nonlinear_ess.png`, `reports/figures/lgssm_validation_means.png`

## Reproduction Commands
(All CLIs support `--config` with presets under `configs/q2/`; sample invocations below show explicit arguments for clarity.)
- Filters (multi-seed):
  - `python -m mlcoe_q2.pipelines.run_multiseed_benchmarks --config configs/q2/run_multiseed_benchmarks.json`
- PF-PF (Stochastic vs LEDH, multi-seed):
  - `python -m mlcoe_q2.pipelines.run_pfpf_stochastic_multiseed --config configs/q2/run_pfpf_stochastic_multiseed.json`
- Li (2017) plots (Kernel flows):
  - `python -m mlcoe_q2.pipelines.plot_flow_diagnostics --config configs/q2/plot_flow_diagnostics.json`
- Hu (2021) high-D Kernel plots:
  - `python -m mlcoe_q2.pipelines.plot_highdim_kernel_flows --config configs/q2/plot_highdim_kernel_flows.json`
- Li (2017) PF-PF reproduction (EDH vs LEDH):
  - `python -m mlcoe_q2.pipelines.reproduce_li17_pfpf --config configs/q2/reproduce_li17_pfpf.json`
- DPF resampling comparisons (soft vs OT_low vs OT):
  - `python -m mlcoe_q2.pipelines.run_dpf_comparisons_multiseed --config configs/q2/run_dpf_comparisons_multiseed.json`
- PF-PF Dai (2022) parameter sweep:
  - `python -m mlcoe_q2.pipelines.run_pfpf_dai22_multiseed --config configs/q2/run_pfpf_dai22_multiseed.json`
- PMMH vs HMC bonus comparison:
  - `python -m mlcoe_q2.pipelines.pmmh_vs_hmc_dpf --config configs/q2/pmmh_vs_hmc_dpf.json`
- Neural OT accelerator:
  - `python -m mlcoe_q2.pipelines.neural_ot_acceleration --config configs/q2/neural_ot_acceleration.json`
- Neural state-space inference:
  - `python -m mlcoe_q2.pipelines.neural_state_space_inference --config configs/q2/neural_state_space_inference.json`
- Literature review notes:
  - `reports/q2/notes/literature_review.md`

For CPU-friendly smoke tests, swap the configs above with the presets under
`configs/q2/quickstart/` (documented in `reports/q2/quickstart.md`).
