"""Compare LLM response metrics against structured forecaster evaluation slices."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


FORECASTER_REQUIRED_COLUMNS = {
    "ticker",
    "target_period",
    "mode",
    "assets_mae",
    "equity_mae",
    "identity_gap",
}
LLM_REQUIRED_COLUMNS = {
    "ticker",
    "context_period",
    "target_period",
    "mae",
    "mape",
    "coverage",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the comparison CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecaster-eval",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reports/q1/artifacts/forecaster_evaluation.parquet",
        help="Parquet artifact emitted by mlcoe_q1.pipelines.evaluate_forecaster",
    )
    parser.add_argument(
        "--llm-metrics",
        type=Path,
        required=True,
        help="Table of per-record metrics from evaluate_llm_responses",
    )
    parser.add_argument(
        "--forecaster-mode",
        default="mlp",
        help="Structured evaluation mode to retain (use 'all' to keep every mode)",
    )
    parser.add_argument(
        "--llm-model",
        help="Optional model identifier to filter inside the LLM metrics table",
    )
    parser.add_argument(
        "--llm-model-column",
        default="model",
        help="Column containing model identifiers inside the LLM metrics table",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=["ticker"],
        help="Columns used to aggregate the joined comparison table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to persist the per-record comparison (csv/json/parquet)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional path to persist the grouped summary table",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity")
    return parser.parse_args(argv)


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_forecaster_evaluation(path: Path, mode: str) -> pd.DataFrame:
    """Load the forecaster evaluation artifact and optionally filter by mode."""

    df = pd.read_parquet(path)
    _ensure_columns(df, FORECASTER_REQUIRED_COLUMNS)
    if 'target_period' in df.columns:
        df['target_period'] = pd.to_datetime(df['target_period'])
    if 'prev_period' in df.columns:
        df['prev_period'] = pd.to_datetime(df['prev_period'])

    if mode.lower() != "all":
        filtered = df[df["mode"].str.lower() == mode.lower()]
        if filtered.empty:
            raise ValueError(f"No forecaster rows remain after filtering for mode={mode!r}")
        df = filtered

    columns = [
        "ticker",
        "target_period",
        "mode",
        "assets_mae",
        "equity_mae",
        "identity_gap",
    ]
    if "net_income_mae" in df.columns:
        columns.append("net_income_mae")
    if "prev_period" in df.columns:
        columns.append("prev_period")

    return df[columns].reset_index(drop=True)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format for {path}")


def load_llm_metrics(path: Path, model_column: str, model: str | None) -> pd.DataFrame:
    """Load and optionally filter LLM evaluation metrics."""

    df = _read_table(path)
    _ensure_columns(df, LLM_REQUIRED_COLUMNS)
    if 'target_period' in df.columns:
        df['target_period'] = pd.to_datetime(df['target_period'])
    if 'context_period' in df.columns:
        df['context_period'] = pd.to_datetime(df['context_period'])

    if model and model_column not in df.columns:
        raise KeyError(
            f"LLM metrics table missing model column {model_column!r} required for filtering"
        )

    if model and model_column in df.columns:
        filtered = df[df[model_column] == model]
        if filtered.empty:
            raise ValueError(
                f"No LLM metrics remain after filtering for {model_column}={model!r}"
            )
        df = filtered

    columns = [
        "ticker",
        "context_period",
        "target_period",
        "mae",
        "mape",
        "coverage",
    ]
    optional_columns = [col for col in ("missing_items", "extra_items", "invalid_items") if col in df.columns]
    columns.extend(optional_columns)
    if model_column in df.columns and model_column not in columns:
        columns.append(model_column)

    return df[columns].reset_index(drop=True)


def compare_metrics(
    forecaster_df: pd.DataFrame, llm_df: pd.DataFrame
) -> pd.DataFrame:
    """Join structured and LLM metrics on matching ticker / period slices."""

    forecaster_renamed = forecaster_df.rename(
        columns={
            "assets_mae": "forecaster_assets_mae",
            "equity_mae": "forecaster_equity_mae",
            "net_income_mae": "forecaster_net_income_mae",
            "identity_gap": "forecaster_identity_gap",
        }
    )

    llm_renamed = llm_df.rename(
        columns={
            "mae": "llm_mae",
            "mape": "llm_mape",
            "coverage": "llm_coverage",
        }
    ).copy()
    for column in ["llm_mae", "llm_mape", "llm_coverage"]:
        if column in llm_renamed.columns:
            llm_renamed[column] = pd.to_numeric(llm_renamed[column], errors="coerce")

    merged = pd.merge(
        llm_renamed,
        forecaster_renamed,
        on=["ticker", "target_period"],
        how="inner",
        suffixes=("_llm", "_forecaster"),
    )

    if merged.empty:
        raise ValueError("No overlapping records between LLM metrics and forecaster evaluation")

    ordered_columns = [
        "ticker",
        "context_period",
        "target_period",
        "llm_mae",
        "llm_mape",
        "llm_coverage",
    ]
    for optional in ["missing_items", "extra_items", "invalid_items"]:
        if optional in merged.columns:
            ordered_columns.append(optional)
    ordered_columns.extend([
        "forecaster_mode" if "mode" not in ordered_columns else "mode",
    ])
    if "mode" in merged.columns:
        merged = merged.rename(columns={"mode": "forecaster_mode"})
    if "prev_period" in merged.columns:
        ordered_columns.append("prev_period")
    ordered_columns.extend(
        [
            "forecaster_assets_mae",
            "forecaster_equity_mae",
            "forecaster_identity_gap",
        ]
    )
    if "forecaster_net_income_mae" in merged.columns:
        ordered_columns.append("forecaster_net_income_mae")

    extra_columns = [col for col in merged.columns if col not in ordered_columns]
    ordered_columns.extend(extra_columns)

    available = [col for col in ordered_columns if col in merged.columns]
    return merged[available].sort_values(["ticker", "target_period"]).reset_index(drop=True)


def summarize_comparison(df: pd.DataFrame, group_by: Sequence[str]) -> pd.DataFrame:
    """Aggregate the comparison table using simple mean statistics."""

    if df.empty:
        return pd.DataFrame()

    if group_by:
        grouped = df.groupby(list(group_by), dropna=False)
        rows = []
        for key, frame in grouped:
            row = _summarize_frame(frame)
            if isinstance(key, tuple):
                for idx, column in enumerate(group_by):
                    row[column] = key[idx]
            else:
                row[group_by[0]] = key
            rows.append(row)
        columns = list(group_by) + [c for c in rows[0].keys() if c not in group_by]
        return pd.DataFrame(rows)[columns]

    return pd.DataFrame([_summarize_frame(df)])


def _summarize_frame(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "records": int(len(frame)),
        "llm_mae_mean": float(frame["llm_mae"].mean(skipna=True)),
        "llm_mape_mean": float(frame["llm_mape"].mean(skipna=True)),
        "llm_coverage_mean": float(frame["llm_coverage"].fillna(0.0).mean()),
        "forecaster_assets_mae_mean": float(frame["forecaster_assets_mae"].mean()),
        "forecaster_equity_mae_mean": float(frame["forecaster_equity_mae"].mean()),
        "forecaster_identity_gap_mean": float(frame["forecaster_identity_gap"].mean()),
    }
    if "forecaster_net_income_mae" in frame.columns:
        summary["forecaster_net_income_mae_mean"] = float(
            frame["forecaster_net_income_mae"].mean()
        )
    return summary


def _write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.fillna("N/A").to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix == ".json":
        df.fillna("N/A").to_json(path, orient="records", indent=2)
    else:
        raise ValueError(f"Unsupported output format for {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.forecaster_eval.exists():
        raise FileNotFoundError(f"Forecaster evaluation not found: {args.forecaster_eval}")
    if not args.llm_metrics.exists():
        raise FileNotFoundError(f"LLM metrics table not found: {args.llm_metrics}")

    forecaster_df = load_forecaster_evaluation(args.forecaster_eval, args.forecaster_mode)
    llm_df = load_llm_metrics(args.llm_metrics, args.llm_model_column, args.llm_model)
    comparison = compare_metrics(forecaster_df, llm_df)

    summary = summarize_comparison(comparison, args.group_by)

    if summary.empty:
        logging.info("No overlapping records detected between the evaluation sources")
    else:
        try:
            rendered = summary.to_markdown(index=False)
        except (ImportError, ModuleNotFoundError):
            rendered = summary.to_string(index=False)
        logging.info("Comparison summary\n%s", rendered)

    if args.output:
        _write_output(comparison, args.output)
        logging.info("Per-record comparison written to %s", args.output)

    if args.summary_output:
        _write_output(summary, args.summary_output)
        logging.info("Grouped summary written to %s", args.summary_output)


if __name__ == "__main__":  # pragma: no cover
    main()

