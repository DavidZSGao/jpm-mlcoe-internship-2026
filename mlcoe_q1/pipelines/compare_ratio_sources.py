"""Compare structured data ratios with PDF-extracted ratios for a given ticker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

STRUCTURED_KEYS = [
    'net_income',
    'cost_to_income_ratio',
    'quick_ratio',
    'debt_to_equity',
    'debt_to_assets',
    'debt_to_capital',
    'debt_to_ebitda',
    'interest_coverage',
]

PDF_KEYS = {
    'net_income': 'net_income',
    'cost_to_income_ratio': 'cost_to_income_ratio',
    'quick_ratio': 'quick_ratio',
    'debt_to_equity': 'debt_to_equity',
    'debt_to_assets': 'debt_to_assets',
    'debt_to_capital': 'debt_to_capital',
    'debt_to_ebitda': 'debt_to_ebit',
    'interest_coverage': 'interest_coverage',
}


def load_json(path: Path) -> Dict[str, float]:
    with open(path) as fh:
        return json.load(fh)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--structured', type=Path, required=True, help='Structured JSON ratios (from compute_ratios).')
    parser.add_argument('--pdf', type=Path, required=True, help='PDF-extracted ratios JSON.')
    parser.add_argument('--output', type=Path, default=Path('reports/q1/artifacts/ratio_comparison.parquet'))
    parser.add_argument('--ticker', default='GM')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    structured = load_json(args.structured)
    pdf_ratios = load_json(args.pdf)

    records = []
    for key in STRUCTURED_KEYS:
        structured_value = structured.get(key)
        pdf_value = pdf_ratios.get(PDF_KEYS[key])
        if structured_value is None and pdf_value is None:
            continue
        records.append(
            {
                'ticker': args.ticker.upper(),
                'metric': key,
                'structured_value': structured_value,
                'pdf_value': pdf_value,
                'delta': None if structured_value is None or pdf_value is None else structured_value - pdf_value,
            }
        )

    df = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)


if __name__ == '__main__':
    main()
