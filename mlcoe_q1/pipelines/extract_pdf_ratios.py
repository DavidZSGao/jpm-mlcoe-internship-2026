"""Extract financial ratios from PDFs using configurable label patterns.

This module was originally sketched as a quick script for the GM annual
report.  As we scale Question 1 coverage to other issuers we need better
observability and configuration hygiene so extractions are reproducible.

Enhancements in this revision:

* both income statement and balance sheet extractors return metadata
  describing which page/strategy succeeded (useful when debugging
  layout-specific failures);
* the CLI can load bespoke configs via ``--config`` while still shipping a
  reasonable built-in default set for GM, LVMH, Tencent, and Alibaba;
* the JSON artifact now records provenance (timestamp, pdfplumber version,
  source PDF, company key) alongside ratios.

This keeps the extraction deterministic while giving downstream analyses a
clear audit trail – an explicit action item from the Question 1 roadmap.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pdfplumber

COMPANY_CONFIG: Dict[str, Dict] = {
    'exxon': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': [
                'Total revenues and other income',
                'Total revenues',
                'Revenues',
            ],
            'operating_income': [
                'Income from operations',
                'Earnings from operations',
                'Operating income',
            ],
            'net_income': [
                'Net income',
                'Net income attributable to ExxonMobil',
                'Net income attributable to Exxon Mobil Corporation',
            ],
            'interest_labels': [
                'Debt-related interest',
                'Interest expense',
                'Interest and debt expense',
            ],
            'expense_mode': 'label',
            'expenses_label': [
                'Total costs and other deductions',
                'Total costs and expenses',
            ],
            'table_markers': [
                'Consolidated statement of income',
                'Consolidated statement of earnings',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total ExxonMobil shareholders’ equity',
                "Total ExxonMobil shareholders' equity",
                'Total equity',
            ],
            'short_term_debt': [
                'Notes and loans payable',
                'Short-term borrowings',
            ],
            'short_term_rows': None,
            'long_term_debt': ['Long-term debt'],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheet',
                'Consolidated statement of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'gm': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Total net sales and revenue'],
            'operating_income': ['Operating income'],
            'net_income': ['Net income.*attributable to stockholders'],
            'interest_labels': ['Automotive interest expense'],
            'expense_mode': 'label',
            'expenses_label': ['Total costs and expenses'],
            'table_markers': ['Total net sales and revenue'],
        },
        'balance': {
            'current_assets': ['total current assets', 'total current ass'],
            'inventory': ['inventories'],
            'total_assets': ['total assets', 'otal assets'],
            'current_liabilities': ['total current liabilities', 'total current liab'],
            'total_liabilities': ['total liabilities', 'otal liabilities'],
            'equity': ['total stockholders’ equity', 'total equity', 'otal equity'],
            'short_term_debt': ['Short-term debt'],
            'short_term_rows': ['automotive', 'gm financial'],
            'long_term_debt': ['Long-term debt'],
            'long_term_rows': ['automotive', 'gm financial'],
            'table_markers': ['Total Assets'],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'}
        ],
    },
    'jpm': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Total net revenue', 'Net revenue'],
            'operating_income': [
                'Income before income tax expense',
                'Income before income tax expense/(benefit)',
            ],
            'net_income': [
                'Net income',
                'Net income/(loss)',
                'Net income attributable to JPMorgan Chase & Co.',
            ],
            'interest_labels': [
                'Interest expense',
                'Interest expense on deposits',
                'Interest expense on long-term debt',
            ],
            'expense_mode': 'label',
            'expenses_label': [
                'Total noninterest expense',
                'Total noninterest expenses',
            ],
            'table_markers': [
                'Consolidated statements of income',
                'Consolidated statements of earnings',
            ],
        },
        'balance': {
            'current_assets': None,
            'inventory': None,
            'total_assets': ['Total assets'],
            'current_liabilities': None,
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total stockholders’ equity',
                "Total stockholders' equity",
                'Total JPMorgan Chase & Co. stockholders’ equity',
                "Total JPMorgan Chase & Co. stockholders' equity",
            ],
            'short_term_debt': [
                'Short-term borrowings',
                'Federal funds purchased',
                'Securities loaned or sold under repurchase agreements',
            ],
            'short_term_rows': None,
            'long_term_debt': ['Long-term debt'],
            'long_term_rows': None,
            'table_markers': ['Consolidated balance sheets'],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'hsbc': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': [
                'Net operating income',
                'Total operating income',
                'Reported revenue',
            ],
            'operating_income': [
                'Profit before tax',
                'Profit before income tax expense',
                'Profit before income tax',
            ],
            'net_income': [
                'Profit attributable to ordinary shareholders',
                'Profit attributable to shareholders of the parent company',
                'Profit for the year',
            ],
            'interest_labels': [
                'Net interest income',
                'Interest expense',
                'Interest expense on financial liabilities',
            ],
            'expense_mode': 'label',
            'expenses_label': [
                'Total operating expenses',
                'Operating expenses',
                'Total operating expense',
            ],
            'table_markers': [
                'Consolidated income statement',
                'Consolidated statement of profit or loss',
            ],
        },
        'balance': {
            'current_assets': None,
            'inventory': None,
            'total_assets': ['Total assets'],
            'current_liabilities': None,
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total shareholders’ equity',
                "Total shareholders' equity",
                'Total equity attributable to shareholders of the parent',
            ],
            'short_term_debt': [
                'Trading liabilities',
                'Deposits by banks',
                'Deposits by customers',
            ],
            'short_term_rows': None,
            'long_term_debt': [
                'Subordinated liabilities',
                'Debt securities in issue',
                'Long-term debt',
            ],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheet',
                'Consolidated statement of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'lvmh': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Revenue'],
            'operating_income': ['Operating profit'],
            'net_income': ['Net profit.*Group share'],
            'interest_labels': ['Cost of net financial debt', 'Interest on lease liabilities'],
            'expense_mode': 'revenue_minus_profit',
            'profit_label': ['Profit from recurring operations'],
            'table_markers': ['Consolidated income statement'],
        },
        'balance': {
            'current_assets': ['Current assets'],
            'inventory': ['Inventories and work in progress'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Current liabilities'],
            'total_liabilities': None,
            'equity': ['Equity'],
            'short_term_debt': ['Short-term borrowings'],
            'short_term_rows': None,
            'long_term_debt': ['Long-term borrowings'],
            'long_term_rows': None,
            'table_markers': ['Consolidated balance sheet'],
        },
        'table_strategies': [
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'}
        ],
    },
    'santander': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': [
                'Total income',
                'Gross income',
                'Net operating income',
            ],
            'operating_income': [
                'Profit before tax',
                'Profit before income tax',
                'Profit before income tax expense',
            ],
            'net_income': [
                'Net profit',
                'Net profit attributable to the Group',
                'Profit attributable to owners of the parent',
            ],
            'interest_labels': [
                'Net interest income',
                'Interest expense',
                'Interest on financial liabilities',
            ],
            'expense_mode': 'label',
            'expenses_label': [
                'Total operating expenses',
                'Operating expenses',
                'Administrative expenses',
            ],
            'table_markers': [
                'Consolidated income statement',
                'Consolidated statement of profit or loss',
            ],
        },
        'balance': {
            'current_assets': None,
            'inventory': None,
            'total_assets': ['Total assets'],
            'current_liabilities': None,
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total equity',
                'Total shareholders’ equity',
                "Total shareholders' equity",
            ],
            'short_term_debt': [
                'Deposits from customers',
                'Deposits from credit institutions',
                'Short-term borrowings',
            ],
            'short_term_rows': None,
            'long_term_debt': [
                'Debt instruments in issue',
                'Senior debt securities',
                'Subordinated liabilities',
            ],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheet',
                'Consolidated statement of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'microsoft': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Revenue', 'Total revenue'],
            'operating_income': ['Operating income', 'Operating income (loss)'],
            'net_income': ['Net income'],
            'interest_labels': ['Interest expense', 'Interest and other, net'],
            'expense_mode': 'label',
            'expenses_label': ['Total costs and expenses', 'Total operating expenses'],
            'table_markers': ['Consolidated statements of income'],
        },
        'balance': {
            'current_assets': ['Total current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total stockholders’ equity',
                "Total stockholders' equity",
            ],
            'short_term_debt': ['Short-term debt'],
            'short_term_rows': None,
            'long_term_debt': ['Long-term debt'],
            'long_term_rows': None,
            'table_markers': ['Consolidated balance sheets'],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'mercedes': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': [
                'Revenue',
                'Total revenue',
                'Revenue and other operating income',
            ],
            'operating_income': [
                'Group EBIT',
                'Earnings before interest and taxes (EBIT)',
                'Operating profit',
            ],
            'net_income': [
                'Net profit',
                'Profit after taxes',
                'Net profit attributable to shareholders of Mercedes-Benz Group',
            ],
            'interest_labels': [
                'Interest expense',
                'Finance costs',
            ],
            'expense_mode': 'label',
            'expenses_label': [
                'Cost of sales',
                'Total expenses',
                'Total cost of sales',
            ],
            'table_markers': [
                'Consolidated statement of income',
                'Consolidated income statement',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total equity',
                'Equity attributable to shareholders of Mercedes-Benz Group',
            ],
            'short_term_debt': [
                'Current financial liabilities',
                'Short-term financial liabilities',
            ],
            'short_term_rows': None,
            'long_term_debt': [
                'Non-current financial liabilities',
                'Long-term financial liabilities',
            ],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated statement of financial position',
                'Consolidated balance sheet',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'nestle': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Sales', 'Total sales', 'Total revenue'],
            'operating_income': [
                'Trading operating profit',
                'Operating profit',
                'Operating income',
            ],
            'net_income': [
                'Net profit',
                'Net profit for the year',
                'Net profit attributable to shareholders of the parent',
            ],
            'interest_labels': [
                'Net financial income/(expense)',
                'Financial income/(expense)',
                'Finance costs',
            ],
            'expense_mode': 'revenue_minus_profit',
            'profit_label': [
                'Trading operating profit',
                'Operating profit',
            ],
            'table_markers': [
                'Consolidated income statement',
                'Consolidated statement of income',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Equity attributable to shareholders of the parent',
                'Total equity',
            ],
            'short_term_debt': [
                'Short-term financial debt',
                'Current financial debt',
            ],
            'short_term_rows': None,
            'long_term_debt': [
                'Long-term financial debt',
                'Non-current financial debt',
            ],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated statement of financial position',
                'Consolidated balance sheet',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'tencent': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Total revenue', 'Revenue'],
            'operating_income': ['Operating profit', 'Operating profit/(loss)'],
            'net_income': ['Profit for the year', 'Profit attributable to equity holders'],
            'interest_labels': [
                'Finance costs',
                'Interest expenses',
                'Interest expense',
                'Finance cost',
            ],
            'expense_mode': 'revenue_minus_profit',
            'profit_label': ['Operating profit', 'Operating profit/(loss)'],
            'table_markers': [
                'Consolidated income statement',
                'Consolidated statement of profit or loss',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets', 'Current assets'],
            'inventory': ['Inventories', 'Inventory'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities', 'Current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total equity',
                'Equity attributable to equity holders of the company',
                'Total equity attributable to equity holders of the company',
            ],
            'short_term_debt': [
                'Bank loans and overdrafts',
                'Current borrowings',
                'Borrowings – current',
                'Borrowings (current)',
            ],
            'short_term_rows': None,
            'long_term_debt': [
                'Non-current borrowings',
                'Borrowings – non-current',
                'Borrowings (non-current)',
                'Notes payable',
            ],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheet',
                'Consolidated statement of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'alibaba': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Total revenue', 'Revenue'],
            'operating_income': [
                'Income from operations',
                'Income from operations/(loss)',
                'Operating income',
            ],
            'net_income': [
                'Net income',
                'Net income attributable to ordinary shareholders of Alibaba Group',
                'Profit for the year attributable to ordinary shareholders of the company',
            ],
            'interest_labels': [
                'Interest expense',
                'Interest and investment income/(loss)',
                'Interest income',
            ],
            'expense_mode': 'revenue_minus_profit',
            'profit_label': [
                'Income from operations',
                'Income from operations/(loss)',
                'Operating income',
            ],
            'table_markers': [
                'Consolidated statements of income',
                'Consolidated statements of operations',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets', 'Current assets'],
            'inventory': ['Inventories', 'Inventory'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities', 'Current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total equity attributable to shareholders of the company',
                'Total shareholders’ equity',
                'Total shareholders\' equity',
            ],
            'short_term_debt': [
                'Short-term borrowings',
                'Current borrowings',
                'Current bank borrowings',
            ],
            'short_term_rows': None,
            'long_term_debt': [
                'Long-term borrowings',
                'Non-current bank borrowings',
                'Non-current borrowings',
            ],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheets',
                'Consolidated statements of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'sap': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Total revenue', 'Revenue', 'Cloud revenue'],
            'operating_income': [
                'Operating profit',
                'Operating income',
                'Profit from operations (EBIT)',
            ],
            'net_income': [
                'Profit after tax',
                'Net income',
                'Profit attributable to owners of parent',
            ],
            'interest_labels': [
                'Finance costs',
                'Interest expense',
            ],
            'expense_mode': 'label',
            'expenses_label': [
                'Total operating expenses',
                'Operating expenses',
            ],
            'table_markers': [
                'Consolidated income statements',
                'Consolidated statement of income',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total equity',
                'Equity attributable to owners of parent',
            ],
            'short_term_debt': [
                'Current financial liabilities',
                'Short-term debt',
            ],
            'short_term_rows': None,
            'long_term_debt': ['Non-current financial liabilities', 'Long-term debt'],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated statements of financial position',
                'Consolidated balance sheets',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'toyota': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Net sales', 'Sales revenue', 'Revenue'],
            'operating_income': [
                'Operating income',
                'Operating profit',
                'Operating income/(loss)',
            ],
            'net_income': [
                'Net income',
                'Net income attributable to owners of the parent',
                'Profit attributable to owners of the parent',
            ],
            'interest_labels': [
                'Interest expense',
                'Interest expenses',
                'Finance costs',
            ],
            'expense_mode': 'revenue_minus_profit',
            'profit_label': [
                'Operating income',
                'Operating profit',
            ],
            'table_markers': [
                'Consolidated statement of income',
                'Consolidated statements of income',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total equity',
                'Total Toyota Motor Corporation shareholders’ equity',
                "Total Toyota Motor Corporation shareholders' equity",
            ],
            'short_term_debt': [
                'Short-term borrowings',
                'Short-term debt',
            ],
            'short_term_rows': None,
            'long_term_debt': [
                'Long-term debt',
                'Long-term borrowings',
            ],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheet',
                'Consolidated statements of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'vw': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Sales revenue', 'Revenue'],
            'operating_income': ['Operating result', 'Operating profit'],
            'net_income': [
                'Profit after tax',
                'Profit attributable to shareholders of Volkswagen AG',
            ],
            'interest_labels': ['Finance costs', 'Interest expense'],
            'expense_mode': 'revenue_minus_profit',
            'profit_label': ['Operating result', 'Operating profit'],
            'table_markers': [
                'Consolidated income statement',
                'Consolidated statement of profit or loss',
            ],
        },
        'balance': {
            'current_assets': ['Current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Equity attributable to shareholders of Volkswagen AG',
                'Total equity',
            ],
            'short_term_debt': [
                'Short-term borrowings',
                'Liabilities to banks (current)',
            ],
            'short_term_rows': None,
            'long_term_debt': [
                'Non-current financial liabilities',
                'Long-term borrowings',
            ],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheet',
                'Consolidated statement of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'alphabet': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Total revenues', 'Revenue'],
            'operating_income': ['Income from operations', 'Operating income'],
            'net_income': [
                'Net income',
                'Net income attributable to Alphabet Inc.',
            ],
            'interest_labels': ['Interest expense', 'Other income (expense), net'],
            'expense_mode': 'revenue_minus_profit',
            'profit_label': ['Operating income', 'Income from operations'],
            'table_markers': [
                'Consolidated statements of income',
                'Consolidated statements of operations',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total stockholders’ equity',
                "Total stockholders' equity",
            ],
            'short_term_debt': ['Current portion of long-term debt', 'Short-term debt'],
            'short_term_rows': None,
            'long_term_debt': ['Long-term debt', 'Non-current debt'],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheets',
                'Consolidated statements of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
    'google': {
        'scale': 1_000_000.0,
        'income': {
            'revenue': ['Total revenues', 'Revenue'],
            'operating_income': ['Income from operations', 'Operating income'],
            'net_income': [
                'Net income',
                'Net income attributable to Alphabet Inc.',
            ],
            'interest_labels': ['Interest expense', 'Other income (expense), net'],
            'expense_mode': 'revenue_minus_profit',
            'profit_label': ['Operating income', 'Income from operations'],
            'table_markers': [
                'Consolidated statements of income',
                'Consolidated statements of operations',
            ],
        },
        'balance': {
            'current_assets': ['Total current assets'],
            'inventory': ['Inventories'],
            'total_assets': ['Total assets'],
            'current_liabilities': ['Total current liabilities'],
            'total_liabilities': ['Total liabilities'],
            'equity': [
                'Total stockholders’ equity',
                "Total stockholders' equity",
            ],
            'short_term_debt': ['Current portion of long-term debt', 'Short-term debt'],
            'short_term_rows': None,
            'long_term_debt': ['Long-term debt', 'Non-current debt'],
            'long_term_rows': None,
            'table_markers': [
                'Consolidated balance sheets',
                'Consolidated statements of financial position',
            ],
        },
        'table_strategies': [
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
        ],
    },
}

Metadata = Dict[str, object]


def parse_value(token: Optional[str], scale: float) -> Optional[float]:
    if token is None:
        return None
    token = str(token).strip().replace(',', '')
    if not token:
        return None
    sign = 1.0
    if token.startswith('(') and token.endswith(')'):
        sign = -1.0
        token = token[1:-1]
    token = token.replace('$', '')
    try:
        return float(token) * scale * sign
    except ValueError:
        return None


def extract_number(row: Sequence[str], scale: float, preferred_index: Optional[int] = None) -> Optional[float]:
    if preferred_index is not None and 0 <= preferred_index < len(row):
        value = parse_value(row[preferred_index], scale)
        if value is not None:
            return value
    cells = list(row[1:])
    for cell in cells:
        value = parse_value(cell, scale)
        if value is not None:
            return value
    row_text = ' '.join(str(cell or '') for cell in row)
    row_text = re.sub(r'\(Note [^\)]*\)', '', row_text)
    row_text = re.sub(r'Note [0-9A-Za-z ()-]*', '', row_text)
    for match in re.findall(r'\$?\(?-?[0-9,]+\)?', row_text):
        clean = match.replace('$', '').strip()
        if not re.search(r'[0-9]', clean):
            continue
        value = parse_value(clean, scale)
        if value is not None:
            return value
    return None


def _normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


def label_matches(label: str, patterns: Optional[Sequence[str]]) -> bool:
    if not patterns:
        return False
    for pattern in patterns:
        if pattern is None:
            continue
        if any(ch in pattern for ch in '^$.*'):
            if re.search(pattern, label, re.IGNORECASE):
                return True
        else:
            if label.lower().startswith(pattern.lower()):
                return True
    return False


def set_if_missing(store: Dict[str, float], key: str, value: Optional[float]) -> None:
    if value is None:
        return
    store.setdefault(key, value)


def _safe_division(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def extract_income_statement(pdf: pdfplumber.PDF, cfg: Dict) -> Tuple[Dict[str, float], Metadata]:
    income_cfg = cfg['income']
    scale = cfg['scale']
    aggregated: Dict[str, float] = {}
    interest_sum = 0.0
    interest_seen = set()
    profit_metric: Optional[float] = None
    required = {'revenue', 'operating_income', 'net_income'}
    if income_cfg.get('expense_mode') == 'label':
        required.add('total_expenses')

    strategies = cfg.get('table_strategies', [
        {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
        {'vertical_strategy': 'text', 'horizontal_strategy': 'text'}
    ])
    for idx in range(max(0, len(pdf.pages) - 70), len(pdf.pages)):
        tables = []
        seen = set()
        for spec in strategies:
            try:
                extracted = pdf.pages[idx].extract_tables(spec) or []
            except Exception:
                continue
            for table in extracted:
                key = tuple(tuple(str(cell) if cell is not None else '' for cell in (row or [])) for row in (table or []))
                if key in seen:
                    continue
                seen.add(key)
                tables.append(table)
        for table in tables:
            required_row_present = False
            for row in table or []:
                if row and label_matches(str(row[0] or ''), income_cfg.get('revenue')):
                    required_row_present = True
                    break
            if not required_row_present:
                continue
            value_idx: Optional[int] = None
            for row in table or []:
                if not row:
                    continue
                if value_idx is None:
                    for idx_cell, cell in enumerate(row[1:], start=1):
                        if re.search(r'\d{4}', str(cell)):
                            value_idx = idx_cell
                            break
                label = (row[0] or '').strip()
                if not label:
                    continue
                if label_matches(label, income_cfg.get('revenue')):
                    set_if_missing(aggregated, 'revenue', extract_number(row, scale, value_idx))
                elif label_matches(label, income_cfg.get('operating_income')):
                    set_if_missing(aggregated, 'operating_income', extract_number(row, scale, value_idx))
                elif label_matches(label, income_cfg.get('net_income')):
                    set_if_missing(aggregated, 'net_income', extract_number(row, scale, value_idx))
                if label_matches(label, income_cfg.get('interest_labels')):
                    value = extract_number(row, scale, value_idx)
                    if value is not None and label not in interest_seen:
                        interest_sum += abs(value)
                        interest_seen.add(label)
                if income_cfg.get('expense_mode') == 'label' and label_matches(label, income_cfg.get('expenses_label')):
                    set_if_missing(aggregated, 'total_expenses', extract_number(row, scale, value_idx))
                if income_cfg.get('expense_mode') == 'revenue_minus_profit' and label_matches(label, income_cfg.get('profit_label')):
                    profit_metric = extract_number(row, scale, value_idx) or profit_metric
            if income_cfg.get('expense_mode') == 'revenue_minus_profit' and 'revenue' in aggregated and profit_metric is not None:
                aggregated['total_expenses'] = abs(aggregated['revenue'] - profit_metric)
            if required <= aggregated.keys():
                aggregated['interest_expense'] = interest_sum
                return aggregated, {
                    'page_index': idx,
                    'strategy': spec,
                    'table_rows': len(table or []),
                }
    raise RuntimeError('Income statement table not found')


def extract_balance_sheet(pdf: pdfplumber.PDF, cfg: Dict) -> Tuple[Dict[str, float], Metadata]:
    balance_cfg = cfg['balance']
    scale = cfg['scale']
    aggregated: Dict[str, float] = {}

    strategies = cfg.get('table_strategies', [
        {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
        {'vertical_strategy': 'text', 'horizontal_strategy': 'text'}
    ])
    for idx in range(max(0, len(pdf.pages) - 70), len(pdf.pages)):
        tables = []
        seen = set()
        for spec in strategies:
            try:
                extracted = pdf.pages[idx].extract_tables(spec) or []
            except Exception:
                continue
            for table in extracted:
                key = tuple(tuple(str(cell) if cell is not None else '' for cell in (row or [])) for row in (table or []))
                if key in seen:
                    continue
                seen.add(key)
                tables.append(table)
        for table in tables:
            required_row_present = False
            for row in table or []:
                if row and label_matches(str(row[0] or ''), balance_cfg.get('total_assets')):
                    required_row_present = True
                    break
            if not required_row_present:
                continue
            current_section: Optional[str] = None
            value_idx: Optional[int] = None
            for row in table or []:
                if not row:
                    current_section = None
                    continue
                if value_idx is None:
                    for idx_cell, cell in enumerate(row[1:], start=1):
                        if re.search(r'\d{4}', str(cell)):
                            value_idx = idx_cell
                            break
                label = (row[0] or '').strip()
                if not label:
                    current_section = None
                    continue
                if 'Short-term debt and current portion of long-term debt' in label:
                    current_section = 'short_term'
                    continue
                if label.startswith('Long-term debt ('):
                    current_section = 'long_term'
                    continue

                if current_section == 'short_term':
                    rows_cfg = balance_cfg.get('short_term_rows')
                    if rows_cfg and not label_matches(label, rows_cfg):
                        continue
                    value = extract_number(row, scale, value_idx)
                    if value is not None:
                        aggregated['short_term_debt'] = aggregated.get('short_term_debt', 0.0) + value
                    if 'gmfinancial' in _normalize(label):
                        current_section = None
                    continue
                if current_section == 'long_term':
                    rows_cfg = balance_cfg.get('long_term_rows')
                    if rows_cfg and not label_matches(label, rows_cfg):
                        continue
                    value = extract_number(row, scale, value_idx)
                    if value is not None:
                        aggregated['long_term_debt'] = aggregated.get('long_term_debt', 0.0) + value
                    if 'gmfinancial' in _normalize(label):
                        current_section = None
                    continue

                if label_matches(label, balance_cfg.get('current_assets')):
                    set_if_missing(aggregated, 'current_assets', extract_number(row, scale, value_idx))
                elif label_matches(label, balance_cfg.get('inventory')):
                    set_if_missing(aggregated, 'inventory', extract_number(row, scale, value_idx))
                elif label_matches(label, balance_cfg.get('total_assets')):
                    set_if_missing(aggregated, 'total_assets', extract_number(row, scale, value_idx))
                elif label_matches(label, balance_cfg.get('current_liabilities')):
                    set_if_missing(aggregated, 'current_liabilities', extract_number(row, scale, value_idx))
                elif balance_cfg.get('total_liabilities') and label_matches(label, balance_cfg['total_liabilities']) and 'and equity' not in label.lower():
                    set_if_missing(aggregated, 'total_liabilities', extract_number(row, scale, value_idx))
                elif label_matches(label, balance_cfg.get('equity')):
                    set_if_missing(aggregated, 'equity', extract_number(row, scale, value_idx))
                elif label_matches(label, balance_cfg.get('short_term_debt')):
                    value = extract_number(row, scale, value_idx)
                    if value is not None:
                        aggregated['short_term_debt'] = aggregated.get('short_term_debt', 0.0) + value
                elif label_matches(label, balance_cfg.get('long_term_debt')):
                    value = extract_number(row, scale, value_idx)
                    if value is not None:
                        aggregated['long_term_debt'] = aggregated.get('long_term_debt', 0.0) + value
                else:
                    current_section = None
            if {'total_assets', 'equity', 'current_liabilities'} <= aggregated.keys():
                return aggregated, {
                    'page_index': idx,
                    'strategy': spec,
                    'table_rows': len(table or []),
                }
        if 'total_assets' in aggregated:
            marker_active = False
    raise RuntimeError('Balance sheet table not found')


def compute_ratios(pdf_path: Path, company: str) -> Dict[str, object]:
    company = company.lower()
    if company not in COMPANY_CONFIG:
        raise ValueError(f'Unsupported company config: {company}')
    cfg = COMPANY_CONFIG[company]
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        income, income_meta = extract_income_statement(pdf, cfg)
        balance, balance_meta = extract_balance_sheet(pdf, cfg)

    short_debt = balance.get('short_term_debt', 0.0)
    long_debt = balance.get('long_term_debt', 0.0)
    total_debt = short_debt + long_debt
    if 'total_liabilities' not in balance:
        balance['total_liabilities'] = balance['total_assets'] - balance['equity']

    current_assets = balance.get('current_assets')
    current_liabilities = balance.get('current_liabilities')
    quick_ratio: Optional[float] = None
    if current_assets is not None and current_liabilities not in (None, 0):
        quick_ratio = (current_assets - balance.get('inventory', 0.0)) / current_liabilities

    equity = balance.get('equity')
    total_assets = balance.get('total_assets')
    operating_income = income.get('operating_income')
    interest_expense = income.get('interest_expense')
    total_expenses = income.get('total_expenses')
    revenue = income.get('revenue')

    debt_to_capital_denominator = total_debt + (equity or 0.0)
    if debt_to_capital_denominator == 0:
        debt_to_capital = None
    else:
        debt_to_capital = total_debt / debt_to_capital_denominator

    interest_coverage = None
    if interest_expense is not None and interest_expense != 0:
        interest_coverage = _safe_division(operating_income, abs(interest_expense))

    ratios = {
        'net_income': income.get('net_income'),
        'cost_to_income_ratio': _safe_division(total_expenses, revenue),
        'quick_ratio': quick_ratio,
        'debt_to_equity': _safe_division(total_debt, equity),
        'debt_to_assets': _safe_division(total_debt, total_assets),
        'debt_to_capital': debt_to_capital,
        'interest_coverage': interest_coverage,
        'debt_to_ebit': _safe_division(total_debt, operating_income),
    }
    return {
        'ratios': ratios,
        'metadata': {
            'company_key': company,
            'scale': cfg['scale'],
            'page_count': page_count,
            'income_table': income_meta,
            'balance_sheet_table': balance_meta,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path, help='Path to the PDF')
    parser.add_argument('--company', choices=COMPANY_CONFIG.keys(), required=True)
    parser.add_argument('--config', type=Path, help='Optional path to a JSON file overriding company defaults')
    parser.add_argument('--output', type=Path, default=Path('reports/q1/artifacts/pdf_ratios.json'))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg_override: Optional[Dict] = None
    if args.config:
        with open(args.config) as fh:
            cfg_override = json.load(fh)
        COMPANY_CONFIG[args.company] = cfg_override
    result = compute_ratios(args.pdf, args.company)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result['metadata'].update(
        {
            'source_pdf': str(args.pdf),
            'extraction_timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'pdfplumber_version': getattr(pdfplumber, '__version__', 'unknown'),
            'table_strategies': cfg_override.get('table_strategies') if cfg_override else COMPANY_CONFIG[args.company].get('table_strategies'),
        }
    )
    with open(args.output, 'w') as fh:
        json.dump(result, fh, indent=2)


if __name__ == '__main__':
    main()
