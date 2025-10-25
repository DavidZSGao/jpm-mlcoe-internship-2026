# Li (2017) PF-PF reproduction (EDH vs LEDH)

- **EDH**
  - ESS: `reports/figures/li2017_pfpf_EDH_ess.png`
  - |log-J|: `reports/figures/li2017_pfpf_EDH_logj.png`
  - Per-step LL (normalized): `reports/figures/li2017_pfpf_EDH_loglik.png`
- **LEDH**
  - ESS: `reports/figures/li2017_pfpf_LEDH_ess.png`
  - |log-J|: `reports/figures/li2017_pfpf_LEDH_logj.png`
  - Per-step LL (normalized): `reports/figures/li2017_pfpf_LEDH_loglik.png`

## Reproduce

```bash
python -m mlcoe_q2.pipelines.reproduce_li17_pfpf \
  --config configs/q2/reproduce_li17_pfpf.json
```
