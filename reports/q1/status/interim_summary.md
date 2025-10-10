# Question 1 Interim Snapshot

- **Interim report:** `reports/q1/q1_interim_report.md` captures the current Part 1/Part 2 methodology, evaluation metrics, LLM diagnostics, and testing evidence ahead of the November checkpoint.
- **Deterministic results:** Latest `reports/q1/status/forecaster_status.md` summarises assets/equity/net-income MAE plus identity gaps; bank ensembles now keep BAC/JPM/C equity MAE below $0.01$B.
- **LLM benchmarking:** `reports/q1/status/llm_vs_forecaster_t5.csv` records that the baseline `t5-small` run produced 0% numeric coverage, motivating stronger adapters.
- **Recommendations:** `reports/q1/status/cfo_recommendations.md` packages deterministic vs. LLM insights into ticker-level guidance for stakeholders.
- **Testing:** Continuous integration via `pytest tests/mlcoe_q1 -q` covers the new pipelines, ensuring the interim artefacts are reproducible.

_Use this snapshot when answering “where we are now” during status updates; it links directly to the artefacts backing the interim report._
