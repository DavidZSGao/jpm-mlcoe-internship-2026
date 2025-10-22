# Strategic Lending Playbook

This playbook distils the Question 1 implementation into a field guide for analysts who need to refresh the Strategic Lending deliverables, interpret outputs, and hand governance artefacts to reviewers. It assumes the repository has already been set up using the root `README` instructions.

## 1. Data Refresh Checklist

1. **Download or update filings** using the resilient Yahoo Finance pipeline:
   ```bash
   python -m mlcoe_q1.pipelines.download_statements --tickers GM HON CAT UNP AAPL MSFT JPM BAC C
   ```
2. **Regenerate processed statements** (idempotent even if cached data exists):
   ```bash
   python -m mlcoe_q1.pipelines.prepare_processed_data
   ```
3. **Build driver datasets** with documented feature/lag settings:
   ```bash
   python -m mlcoe_q1.pipelines.build_driver_dataset \
       --config configs/q1/orchestrate_lending_workflow.json
   ```
4. **Validate coverage** before modelling:
   ```bash
   python -m mlcoe_q1.pipelines.validate_driver_dataset \
       --drivers mlcoe_q1/data/processed/driver_features.parquet \
       --output reports/q1/artifacts/driver_validation.csv
   ```

## 2. Forecasting Workflows

### 2.1 Training options

| Model family | Command snippet | Notes |
|--------------|-----------------|-------|
| MLP (baseline) | `python -m mlcoe_q1.pipelines.train_forecaster --config configs/q1/orchestrate_lending_workflow.json` | Deterministic backbone + MC dropout.
| GRU sequence model | add `--architecture gru --sequence-length 4` | Uses recurrent dropout and shared scalers.
| Variational driver head | add `--variational --kl-weight 0.01` | Produces latent samples consumed by Monte Carlo evaluation.
| Gaussian output head | add `--gaussian-head` | Trains mean/variance outputs directly.

### 2.2 Evaluation toolkit

1. **Deterministic + stochastic rollouts**
   ```bash
   python -m mlcoe_q1.pipelines.evaluate_forecaster \
       --config configs/q1/orchestrate_lending_workflow.json \
       --horizon 4 --monte-carlo-samples 200
   ```
2. **Summaries and coverage diagnostics**
   ```bash
   python -m mlcoe_q1.pipelines.summarize_forecaster_evaluation \
       --evaluation reports/q1/artifacts/forecaster_eval.parquet \
       --summary-output reports/q1/status/forecaster_summary.parquet
   python -m mlcoe_q1.pipelines.analyze_forecaster_calibration \
       --evaluation reports/q1/artifacts/forecaster_eval.parquet \
       --summary-output reports/q1/status/forecaster_calibration.parquet
   ```
3. **Scenario packaging & reasonableness checks**
   ```bash
   python -m mlcoe_q1.pipelines.package_scenarios \
       --summary reports/q1/status/forecaster_summary.parquet \
       --scenario-output reports/q1/artifacts/forecaster_scenarios.parquet
   python -m mlcoe_q1.pipelines.assess_scenario_reasonableness \
       --scenarios reports/q1/artifacts/forecaster_scenarios.parquet \
       --output reports/q1/status/scenario_reasonableness.parquet
   ```

## 3. Strategic Lending Extensions

### 3.1 Macro overlays
```bash
python -m mlcoe_q1.pipelines.simulate_macro_conditions \
    --scenario-input reports/q1/artifacts/forecaster_scenarios.parquet \
    --config reports/q1/artifacts/macro_scenarios_example.json \
    --output reports/q1/artifacts/macro_conditioned_scenarios.parquet
```

### 3.2 Credit, pricing, and risk warnings
```bash
python -m mlcoe_q1.pipelines.build_credit_rating_dataset --output reports/q1/artifacts/altman_dataset.parquet
python -m mlcoe_q1.pipelines.score_credit_rating --config configs/q1/orchestrate_lending_workflow.json
python -m mlcoe_q1.pipelines.price_loans \
    reports/q1/artifacts/forecaster_scenarios.parquet \
    --credit-dataset reports/q1/artifacts/altman_dataset.parquet \
    --macro-sensitivity configs/q1/loan_pricing_macro.json \
    --summary-output reports/q1/artifacts/loan_pricing_summary.json
python -m mlcoe_q1.pipelines.extract_risk_warnings \
    --input reports/q1/artifacts/llm_runs/prompts.parquet \
    --output reports/q1/artifacts/risk_warnings.parquet
```

### 3.3 PDF extraction roster
- GM, LVMH, Tencent, Alibaba
- JPMorgan Chase, HSBC, Banco Santander
- ExxonMobil, Toyota, Nestlé
- Microsoft, Alphabet/Google, SAP, Mercedes-Benz, Volkswagen

Run with, for example:
```bash
python -m mlcoe_q1.pipelines.extract_pdf_ratios \
    --issuer jpmorgan \
    --pdf-path data/pdfs/jpm_2023.pdf \
    --output reports/q1/artifacts/jpm_ratios.parquet
```

## 4. LLM Benchmarking & Governance

1. **Prompt dataset**
   ```bash
   python -m mlcoe_q1.pipelines.build_llm_prompt_dataset --config configs/q1/run_llm_adapter.json
   ```
2. **Adapter runs with metadata**
   ```bash
   python -m mlcoe_q1.pipelines.run_llm_adapter --config configs/q1/run_llm_adapter.json
   ```
3. **Benchmark suite**
   ```bash
   python -m mlcoe_q1.pipelines.benchmark_llm_suite --config configs/q1/benchmark_llm_suite.json \
       --seed-summary-output reports/q1/artifacts/llm_seed_summary.parquet
   ```
4. **Benchmark summary briefing**
   ```bash
   python -m mlcoe_q1.pipelines.summarize_llm_benchmarks --config configs/q1/summarize_llm_benchmarks.json
   ```
5. **Evaluation + CFO recommendations**
   ```bash
   python -m mlcoe_q1.pipelines.evaluate_llm_responses --config configs/q1/benchmark_llm_suite.json
   python -m mlcoe_q1.pipelines.generate_cfo_recommendations --config configs/q1/benchmark_llm_suite.json
   ```

## 5. Reporting, Audit, and Publishing

0. **Workplan progress snapshot**
   ```bash
   python -m mlcoe_q1.pipelines.summarize_workplan_progress \
       --config configs/q1/summarize_workplan_progress.json
   ```
1. **Strategic Lending briefing**
   ```bash
   python -m mlcoe_q1.pipelines.generate_lending_briefing --config configs/q1/orchestrate_lending_workflow.json
   ```
2. **Executive summary**
   ```bash
   python -m mlcoe_q1.pipelines.generate_executive_summary --config configs/q1/publish_lending_submission.json
   ```
3. **Audit expectations**
   ```bash
   python -m mlcoe_q1.pipelines.audit_lending_artifacts \
       --config configs/q1/lending_audit_expectations.json
   ```
4. **Publish submission bundle**
   ```bash
   python -m mlcoe_q1.pipelines.publish_lending_submission \
       --config configs/q1/publish_lending_submission.json
   ```

## 6. One-Command Orchestration

To refresh everything (evaluation, scenarios, macro overlays, briefings, audit, executive summary, and optional zip bundle) in a single run:
```bash
python -m mlcoe_q1.pipelines.orchestrate_lending_workflow --config configs/q1/orchestrate_lending_workflow.json
python -m mlcoe_q1.pipelines.publish_lending_submission --config configs/q1/publish_lending_submission.json --zip-output reports/q1/deliverables/strategic_lending.zip
```

## 7. Deliverable Inventory

The orchestrator and publisher manifest the following key artefacts:

| Category | Path | Description |
|----------|------|-------------|
| Forecast evaluation | `reports/q1/artifacts/forecaster_eval.parquet` | Per-ticker rollouts with MC samples & coverage diagnostics. |
| Summary metrics | `reports/q1/status/forecaster_summary.parquet` | Grouped MAE/MAPE, interval width, coverage stats. |
| Scenarios | `reports/q1/artifacts/forecaster_scenarios.parquet` | Quantile-point scenario tables consumed by pricing & macro overlays. |
| Macro overlays | `reports/q1/artifacts/macro_conditioned_scenarios.parquet` | Scenario adjustments under configured macro shocks. |
| Credit & pricing | `reports/q1/artifacts/altman_scores.parquet`, `reports/q1/artifacts/loan_pricing.parquet` | Altman Z-scores, recommended loan spreads. |
| LLM benchmarking | `reports/q1/artifacts/llm_runs/` | Responses, metrics, metadata, and seed summary. |
| Risk warnings | `reports/q1/artifacts/risk_warnings.parquet` | Banking risk lexicon hits from structured text. |
| Reporting | `reports/q1/status/strategic_lending_briefing.md`, `reports/q1/status/executive_summary.md` | Analyst-ready Markdown briefings. |
| Governance | `reports/q1/status/lending_audit.md`, `reports/q1/status/lending_audit.json` | Audit results against expectation configs. |
| Deliverable bundle | `reports/q1/deliverables/strategic_lending/`, optional `.zip` | Packaged submission with manifest. |

## 8. Troubleshooting Tips

- **LLM timeouts:** Use `--max-workers` to throttle concurrency and retry with `--resume-from` when available.
- **Hosted API quotas:** Supply `OPENAI_API_KEY` (or relevant provider key) and reduce `--max-concurrent-requests` in the benchmarking config.
- **PDF parsing edge cases:** Add issuer-specific label overrides via `--label-overrides path/to/overrides.json` instead of editing presets.
- **Monte Carlo convergence:** Increase `--monte-carlo-samples` and verify calibration with `analyze_forecaster_calibration` before scenario packaging.
- **Governance diffs:** Re-run `audit_lending_artifacts` after manual edits; expectations live in `configs/q1/lending_audit_expectations.json`.

## 9. Completion Criteria

You can consider the Strategic Lending deliverables fully refreshed when:

- Forecast evaluation, scenario packaging, macro overlays, credit/pricing, LLM benchmarking, and risk warnings have all been regenerated in the current reporting window.
- `publish_lending_submission` completes without audit failures and produces an up-to-date manifest plus optional zip bundle.
- The executive summary references the latest macro scenario name, LLM suite manifest, and audit timestamp (visible in the generated Markdown).

Stay within the provided configs wherever possible so that governance artefacts remain reproducible across refresh cycles.
