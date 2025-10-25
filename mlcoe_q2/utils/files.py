"""File-system helpers for pipeline outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_output_paths(paths: Iterable[Path | None]) -> None:
    """Ensure that parent directories for the provided paths exist."""

    for path in paths:
        if path is None:
            continue
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


__all__ = ["ensure_output_paths"]
