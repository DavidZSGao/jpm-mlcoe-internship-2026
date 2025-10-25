# Li (2017)-Style Flow Diagnostics (Kernel)

- **KernelScalar**
  - Residuals: `reports/figures/li2017_KernelScalar_residuals.png`
  - Movement: `reports/figures/li2017_KernelScalar_movement.png`
  - Log-Jacobian: `reports/figures/li2017_KernelScalar_logjac.png`
- **KernelDiagonal**
  - Residuals: `reports/figures/li2017_KernelDiagonal_residuals.png`
  - Movement: `reports/figures/li2017_KernelDiagonal_movement.png`
  - Log-Jacobian: `reports/figures/li2017_KernelDiagonal_logjac.png`
- **KernelMatrix**
  - Residuals: `reports/figures/li2017_KernelMatrix_residuals.png`
  - Movement: `reports/figures/li2017_KernelMatrix_movement.png`
  - Log-Jacobian: `reports/figures/li2017_KernelMatrix_logjac.png`

## Reproduce

```bash
python -m mlcoe_q2.pipelines.plot_flow_diagnostics \
  --config configs/q2/plot_flow_diagnostics.json
```
