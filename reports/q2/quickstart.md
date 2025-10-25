# Question 2 Quickstart Runs

The full Question 2 pipelines mirror the submission artifacts and therefore favour
statistical stability over turnaround time:

- Multi-seed filter/flow sweeps run **5 seeds × 15–20 steps** with **256 particles**
  per filter, tracking runtime, memory, likelihood, and ESS for each variant.
- The PMMH vs DPF-HMC comparison keeps **200+ samples** in both chains so the
  acceptance and RMSE metrics in the final report are representative.
- Neural extensions (neural OT accelerator and LSTM state-space inference)
  train for **40 epochs** or run **300 DPF-HMC iterations**, which pays the TensorFlow
  compilation overhead but can take multiple hours on a CPU-only workstation.

These defaults align with the deliverables but can feel like "days" of compute on
resource-constrained machines. To iterate quickly or validate an installation,
use the lightweight presets below. Each command reuses the same pipelines but
shrinks particle counts, seeds, and training loops; outputs are written to
`reports/artifacts/quickstart/` and `reports/q2/status/quickstart/` so the
submission evidence is left untouched.

## Lightweight command matrix

| Scope | Command |
| --- | --- |
| Filters vs PF baselines | `python -m mlcoe_q2.pipelines.run_multiseed_benchmarks --config configs/q2/quickstart/run_multiseed_benchmarks.json` |
| Deterministic vs stochastic PF-PF flows | `python -m mlcoe_q2.pipelines.run_pfpf_stochastic_multiseed --config configs/q2/quickstart/run_pfpf_stochastic_multiseed.json` |
| Dai (2022) stochastic flow sweep | `python -m mlcoe_q2.pipelines.run_pfpf_dai22_multiseed --config configs/q2/quickstart/run_pfpf_dai22_multiseed.json` |
| DPF resampling comparison | `python -m mlcoe_q2.pipelines.run_dpf_comparisons_multiseed --config configs/q2/quickstart/run_dpf_comparisons_multiseed.json` |
| PMMH vs DPF-HMC | `python -m mlcoe_q2.pipelines.pmmh_vs_hmc_dpf --config configs/q2/quickstart/pmmh_vs_hmc_dpf.json` |
| Neural OT accelerator training | `python -m mlcoe_q2.pipelines.neural_ot_acceleration --config configs/q2/quickstart/neural_ot_acceleration.json` |
| Neural state-space inference | `python -m mlcoe_q2.pipelines.neural_state_space_inference --config configs/q2/quickstart/neural_state_space_inference.json` |

## What to expect

- Wall-clock time is typically **5–10× faster** than the full submission runs on a
  8-core laptop (tens of minutes instead of many hours).
- Metrics trend in the same direction as the high-fidelity artifacts, but the
  smaller seeds/particles introduce more variance. Use the original configs for
  report-quality numbers.
- The quickstart outputs deliberately live in a separate folder. Delete the
  `reports/artifacts/quickstart/` directory between experiments if you want a
  clean slate.

Refer back to `reports/q2/final_report.md` for the full-scope evidence. When you
are ready to regenerate the submission artifacts, switch back to the baseline
configs in `configs/q2/`.
