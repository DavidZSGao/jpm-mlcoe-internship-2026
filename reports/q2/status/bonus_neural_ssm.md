# Bonus — Neural State-Space Inference

| Method | Acceptance | ESS (log proc) | ESS (log obs) | Runtime (s) | RMSE (log proc) | RMSE (log obs) |
| --- | --- | --- | --- | --- | --- | --- |
| DPF-HMC | 1.00 | 5.4 | 5.9 | 30.12 | 0.306 | 0.280 |
| Particle Gibbs | 0.71 | 3.4 | 8.0 | 8.50 | 0.197 | 0.097 |

## Posterior Means

- DPF-HMC: log process = 0.098, log observation = -0.080
- Particle Gibbs: log process = -0.362, log observation = -0.264

## Reproduce

```bash
python -m mlcoe_q2.pipelines.neural_state_space_inference --config configs/q2/neural_state_space_inference.json
```