"""Build text chunks from annual report PDFs/HTML for risk-warning extraction."""

from __future__ import annotations

import argparse
import html
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from mlcoe_q1.utils.config import add_config_argument, parse_args_with_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_argument(parser)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="PDF/HTML report paths to ingest (same order as --issuers).",
    )
    parser.add_argument(
        "--issuers",
        nargs="+",
        required=True,
        help="Issuer names aligned with --inputs.",
    )
    parser.add_argument(
        "--section",
        default="annual_report",
        help="Section label to attach to every chunk.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Approximate character budget per chunk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports/q1/artifacts/risk_text_chunks.parquet",
        help="Output parquet path for text chunks.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parse_args_with_config(
        parser,
        argv,
        type_overrides={
            "inputs": Path,
            "output": Path,
        },
    )


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_html(raw: str) -> str:
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\\1>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<br\\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</p>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = html.unescape(cleaned)
    return _normalize_text(cleaned)


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            for idx in range(0, len(line), chunk_size):
                piece = line[idx : idx + chunk_size].strip()
                if piece:
                    chunks.append(piece)
            continue
        if current and current_len + len(line) + 1 > chunk_size:
            chunks.append(" ".join(current).strip())
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line) + 1
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _iter_pdf_chunks(path: Path) -> Iterable[tuple[int | None, str]]:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = _normalize_text(page.extract_text() or "")
            if text:
                yield idx, text


def _iter_html_chunks(path: Path, chunk_size: int) -> Iterable[tuple[int | None, str]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = _clean_html(raw)
    for chunk in _chunk_text(cleaned, chunk_size):
        yield None, chunk


def _iter_chunks(path: Path, chunk_size: int) -> Iterable[tuple[int | None, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        yield from _iter_pdf_chunks(path)
    elif suffix in {".html", ".htm"}:
        yield from _iter_html_chunks(path, chunk_size)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def build_chunks(
    inputs: Sequence[Path],
    issuers: Sequence[str],
    section: str,
    chunk_size: int,
) -> pd.DataFrame:
    if len(inputs) != len(issuers):
        raise ValueError("--inputs and --issuers must have the same length")

    records: list[dict[str, object]] = []
    for path, issuer in zip(inputs, issuers):
        for page, text in _iter_chunks(path, chunk_size):
            records.append(
                {
                    "issuer": issuer,
                    "section": section,
                    "page": page,
                    "text": text,
                }
            )
    return pd.DataFrame.from_records(records)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    chunks = build_chunks(args.inputs, args.issuers, args.section, args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(args.output, index=False)
    logging.info("Wrote %d chunks to %s", len(chunks), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
