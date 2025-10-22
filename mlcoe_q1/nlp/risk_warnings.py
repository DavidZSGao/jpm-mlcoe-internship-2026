"""Identify risk-warning disclosures in annual-report text chunks."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

# Keyword lexicon grouped by disclosure category. Patterns use lowercase strings.
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "going_concern": (
        "going concern",
        "substantial doubt",
        "doubt about our ability to continue",
    ),
    "liquidity": (
        "liquidity risk",
        "liquidity constraints",
        "material weakness in liquidity",
        "insufficient cash",
        "refinancing risk",
    ),
    "capital_markets": (
        "downgrade",
        "credit rating agency",
        "covenant breach",
        "debt covenant",
        "maturity wall",
    ),
    "regulatory": (
        "regulatory action",
        "consent order",
        "material non-compliance",
        "regulatory investigation",
    ),
    "operational": (
        "supply chain disruption",
        "cybersecurity incident",
        "operational disruption",
        "system outage",
    ),
    "geopolitical": (
        "sanction risk",
        "geopolitical instability",
        "trade restriction",
    ),
}

DEFAULT_RISK_SEVERITY: Mapping[str, str] = {
    "going_concern": "high",
    "liquidity": "high",
    "capital_markets": "medium",
    "regulatory": "medium",
    "operational": "medium",
    "geopolitical": "medium",
}


def _load_json_records(path: Path) -> list[MutableMapping[str, object]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, Mapping) and "records" in payload:
        raw = payload["records"]
    else:
        raw = payload

    if not isinstance(raw, Sequence):
        raise ValueError("JSON file must contain a list of records or a 'records' field")

    return [dict(item) for item in raw]


def load_text_chunks(path: Path) -> pd.DataFrame:
    """Load structured text chunks from JSON or Parquet into a DataFrame."""

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        records = _load_json_records(path)
        frame = pd.DataFrame.from_records(records)
    else:
        raise ValueError("Unsupported input format. Use .json, .jsonl, or .parquet")

    expected_cols = {"issuer", "section", "text"}
    missing = expected_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["text"] = frame["text"].fillna("")
    frame["section"] = frame["section"].fillna("unknown")
    if "page" not in frame.columns:
        frame["page"] = pd.NA
    return frame[["issuer", "section", "page", "text"]]

def _sentences(text: str) -> Iterable[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    return [part.strip() for part in parts if part.strip()]


def _find_snippet(text: str, keyword: str) -> str:
    for sentence in _sentences(text):
        if keyword.lower() in sentence.lower():
            return sentence
    text = text.strip()
    return text[:240] if len(text) > 240 else text


def extract_risk_warnings(
    frame: pd.DataFrame,
    *,
    category_keywords: Mapping[str, Sequence[str]] | None = None,
    severity_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Identify risk warnings by scanning text for configured keywords."""

    keywords = category_keywords or CATEGORY_KEYWORDS
    severity = severity_map or DEFAULT_RISK_SEVERITY

    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        text = str(row["text"] or "")
        if not text.strip():
            continue
        issuer = row["issuer"]
        section = row["section"]
        page = row.get("page")
        for category, phrases in keywords.items():
            for phrase in phrases:
                if phrase.lower() in text.lower():
                    snippet = _find_snippet(text, phrase)
                    records.append(
                        {
                            "issuer": issuer,
                            "section": section,
                            "page": page,
                            "category": category,
                            "keyword": phrase,
                            "severity": severity.get(category, "medium"),
                            "snippet": snippet,
                        }
                    )
    return pd.DataFrame.from_records(records)


def summarise_risk_warnings(risks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate risk warnings per issuer/category with counts and severity."""

    if risks.empty:
        return pd.DataFrame(
            columns=["issuer", "category", "severity", "warning_count", "sections"]
        )

    grouped = (
        risks.groupby(["issuer", "category", "severity"], dropna=False)
        .agg(
            warning_count=("keyword", "count"),
            sections=("section", lambda values: sorted(set(v for v in values if v))),
        )
        .reset_index()
    )
    return grouped


def write_summary_json(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = summary.to_dict(orient="records")
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": records}, handle, indent=2)


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
