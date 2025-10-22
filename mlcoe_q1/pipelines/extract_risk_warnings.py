"""Extract risk-warning disclosures from annual-report text chunks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence


from mlcoe_q1.nlp.risk_warnings import (
    configure_logging,
    extract_risk_warnings,
    load_text_chunks,
    summarise_risk_warnings,
    write_summary_json,
)
from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON/JSONL/Parquet file containing issuer text chunks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/risk_warnings.parquet",
        help="Destination parquet file for extracted risk warnings",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/risk_warnings_summary.json",
        help="Optional JSON summary file with issuer/category aggregates",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level to use for status messages",
    )
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "input": Path,
            "output": Path,
            "summary": Path,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    logging.info("Loading text chunks from %s", args.input)
    chunks = load_text_chunks(args.input)
    logging.info("Scanning %d chunks for risk disclosures", len(chunks))
    risks = extract_risk_warnings(chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    risks.to_parquet(args.output, index=False)
    logging.info("Wrote %d risk warnings to %s", len(risks), args.output)
    summary = summarise_risk_warnings(risks)
    if args.summary:
        write_summary_json(summary, args.summary)
        logging.info("Wrote summary to %s", args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
