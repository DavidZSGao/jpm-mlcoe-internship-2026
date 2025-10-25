# Cross-Question Coordination Plan

## Active Workstreams
- **Q2 — Particle Flows:** Continue with benchmarking + reporting tasks outlined in `reports/q2/gap_closure_plan.md` (Li 2017 plot reproduction, stochastic flow comparisons, HMC integration experiments).
- **Q1 — Lending Application:** Kick off data ingestion + constraint modelling per `reports/q1/notes/q1_project_plan.md` while aligning reporting conventions.

## Immediate Next Actions
1. Finalise Q2 benchmarking figures (Li 2017 reproduction, stochastic flow vs LEDH matrices) to unblock final write-up integration.
2. Start Q1 data ingestion utilities (`yfinance_ingest.py`) and commit initial processed dataset snapshots for selected tickers.
3. Draft shared reporting template to harmonise narrative structure across Q1 and Q2 deliverables.

## Coordination Notes
- Shared dependencies (TensorFlow, data loaders) should be abstracted to avoid duplication—evaluate whether utilities belong in a common module (`common/` or `mlcoe_shared/`) once requirements stabilise.
- Align artifact naming (`*_summary.json`, `*_fig.png`) so reporting scripts can aggregate across questions.
- Keep large raw datasets out of version control; document fetch commands in each question's README.
- Schedule regular checkpoints (e.g., end-of-day summaries) to track progress against both question plans.
