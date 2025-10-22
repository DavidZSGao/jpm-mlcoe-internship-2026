from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlcoe_q1.nlp import risk_warnings
from mlcoe_q1.pipelines import extract_risk_warnings as risk_cli


def build_sample_chunks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "issuer": "ACME",
                "section": "Risk Factors",
                "page": 12,
                "text": (
                    "There is substantial doubt about our ability to continue as a going concern. "
                    "We also face liquidity constraints if we cannot refinance the maturity wall in 2026."
                ),
            },
            {
                "issuer": "ACME",
                "section": "MD&A",
                "page": 33,
                "text": "We remediated the prior-year liquidity risk and no longer see a downgrade trigger.",
            },
            {
                "issuer": "GlobalBank",
                "section": "Risk Factors",
                "page": 88,
                "text": "Regulatory investigation resulted in a consent order related to operational disruption.",
            },
            {
                "issuer": "GlobalBank",
                "section": "Risk Factors",
                "page": 90,
                "text": "",  # should be ignored
            },
        ]
    )


def test_extract_risk_warnings_detects_categories():
    frame = build_sample_chunks()
    risks = risk_warnings.extract_risk_warnings(frame)
    assert {(row.category, row.keyword) for row in risks.itertuples()} == {
        ("going_concern", "substantial doubt"),
        ("going_concern", "going concern"),
        ("going_concern", "doubt about our ability to continue"),
        ("liquidity", "liquidity constraints"),
        ("capital_markets", "maturity wall"),
        ("liquidity", "liquidity risk"),
        ("capital_markets", "downgrade"),
        ("regulatory", "regulatory investigation"),
        ("regulatory", "consent order"),
        ("operational", "operational disruption"),
    }


def test_summarise_risk_warnings_groups_by_issuer():
    risks = risk_warnings.extract_risk_warnings(build_sample_chunks())
    summary = risk_warnings.summarise_risk_warnings(risks)
    row = summary.loc[(summary["issuer"] == "ACME") & (summary["category"] == "liquidity")]
    assert int(row["warning_count"].iloc[0]) == 2
    assert row["sections"].iloc[0] == ["MD&A", "Risk Factors"]


def test_cli_writes_outputs(tmp_path: Path):
    input_path = tmp_path / "chunks.jsonl"
    records = build_sample_chunks().to_dict(orient="records")
    with input_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    output_path = tmp_path / "risks.parquet"
    summary_path = tmp_path / "summary.json"
    exit_code = risk_cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
            "--log-level",
            "DEBUG",
        ]
    )
    assert exit_code == 0
    risks = pd.read_parquet(output_path)
    assert not risks.empty
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]
