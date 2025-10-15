# Q2 Deliverables Index

## Reports
- `reports/q2_benchmark_report.md`
- `reports/q2_interim_report.md`

## Status Snapshots
- Filters (multi-seed): `reports/q2/status/filter_status.md`
- PF-PF (Stochastic vs LEDH): `reports/q2/status/pfpf_status.md`
- Li (2017)-style Kernel flow plots: `reports/q2/status/li2017_plots.md`
- Hu (2021)-style high-dimensional Kernel plots: `reports/q2/status/hu2021_plots.md`
- Li (2017) PF-PF reproduction (EDH vs LEDH): `reports/q2/status/li2017_pfpf_reproduction.md`

## Key Artifacts
- Filters aggregate: `reports/artifacts/benchmark_filters_multiseed.json`
- Filters per-seed: `reports/artifacts/benchmark_filters_seed_{0..4}.json`
- PF-PF aggregate: `reports/artifacts/pfpf_stochastic_multiseed.json`
- PF-PF per-seed: `reports/artifacts/pfpf_stochastic_seed_{0..4}.json`
- Latest single-run snapshot: `reports/artifacts/benchmark_latest.json`
- Diagnostics: `reports/artifacts/nonlinear_diagnostics.json`
- LGSSM validation summary: `reports/artifacts/lgssm_validation_summary.json`

## Figures
- Kernel flow diagnostics (Li 2017-style):
  - `reports/figures/li2017_KernelScalar_residuals.png`, `li2017_KernelScalar_movement.png`, `li2017_KernelScalar_logjac.png`
  - `reports/figures/li2017_KernelDiagonal_residuals.png`, `li2017_KernelDiagonal_movement.png`, `li2017_KernelDiagonal_logjac.png`
  - `reports/figures/li2017_KernelMatrix_residuals.png`, `li2017_KernelMatrix_movement.png`, `li2017_KernelMatrix_logjac.png`
- Additional: `reports/figures/nonlinear_rmse.png`, `reports/figures/nonlinear_ess.png`, `reports/figures/lgssm_validation_means.png`

## Reproduction Commands
- Filters (multi-seed):
  - `python -m mlcoe_q2.experiments.run_multiseed_benchmarks --seeds 0 1 2 3 4 --num-timesteps 15`
- PF-PF (Stochastic vs LEDH, multi-seed):
  - `python -m mlcoe_q2.experiments.run_pfpf_stochastic_multiseed --seeds 0 1 2 3 4 --num-timesteps 15`
- Li (2017) plots (Kernel flows):
  - `python -m mlcoe_q2.experiments.plot_flow_diagnostics --seeds 0 --num-timesteps 15 --particles 256`
- Hu (2021) high-D Kernel plots:
  - `python -m mlcoe_q2.experiments.plot_highdim_kernel_flows --seeds 0 --num-timesteps 10 --particles 256 --state-dim 16 --obs-dim 4`
- Li (2017) PF-PF reproduction (EDH vs LEDH):
  - `python -m mlcoe_q2.experiments.reproduce_li17_pfpf --seeds 0 1 2 --num-timesteps 15 --particles 256`
