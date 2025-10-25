# Bonus — Neural OT Resampling

| Metric | Value |
| --- | --- |
| Train loss | 0.3623 |
| Validation loss | 0.5215 |
| Test loss | 0.4015 |
| Test MAE | 0.3928 |
| Row normalisation error | 2.0833e-02 |
| Transport L1 error | 2.0429e-01 |
| Per-sample Sinkhorn time (s) | 0.05279 |
| Per-sample neural time (s) | 0.00172 |
| Relative speedup | 30.73× |

## Configuration

- Particles: 8
- State dimension: 4
- Samples: 64
- Hidden units: [48, 24]

## Reproduce

```bash
python -m mlcoe_q2.pipelines.neural_ot_acceleration --config configs/q2/neural_ot_acceleration.json
```