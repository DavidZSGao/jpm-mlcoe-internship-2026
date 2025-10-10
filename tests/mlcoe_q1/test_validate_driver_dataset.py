"""Tests for the driver dataset validation CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlcoe_q1.pipelines.validate_driver_dataset import (
    REQUIRED_FEATURES,
    summarise_drivers,
    main as validate_main,
)


def _make_base_frame() -> pd.DataFrame:
    data = {
        "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "CCC", "CCC", "CCC"],
        "period": [
            "2020-12-31",
            "2021-12-31",
            "2022-12-31",
            "2019-12-31",
            "2021-12-31",  # intentional two-year gap
            "2020-12-31",
            "2020-12-31",  # duplicate period
            "2021-12-31",
        ],
    }
    for feature in REQUIRED_FEATURES:
        if feature == "sales":
            data[feature] = [1e9, 1.2e9, 1.3e9, 5e8, 6e8, 4e8, 4e8, None]
        else:
            data[feature] = [0.1] * len(data["ticker"])
    return pd.DataFrame(data)


def test_summarise_drivers_flags_duplicates_and_gaps() -> None:
    df = _make_base_frame()
    summary = summarise_drivers(df, REQUIRED_FEATURES, min_observations=3, max_gap_days=500)

    assert set(summary["ticker"].tolist()) == {"AAA", "BBB", "CCC"}

    aaa_row = summary[summary["ticker"] == "AAA"].iloc[0]
    assert aaa_row["status"] == "ok"
    assert pytest.approx(aaa_row["median_gap_days"], rel=1e-3) == 365

    bbb_row = summary[summary["ticker"] == "BBB"].iloc[0]
    assert bool(bbb_row["long_gap"]) is True
    assert bbb_row["status"] == "needs_attention"

    ccc_row = summary[summary["ticker"] == "CCC"].iloc[0]
    assert bool(ccc_row["duplicate_periods"]) is True
    assert ccc_row["na_rows"] == 1
    assert ccc_row["status"] == "needs_attention"


def test_cli_writes_summary_and_exits_on_issues(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    df = _make_base_frame()
    drivers_path = tmp_path / "drivers.parquet"
    df.to_parquet(drivers_path, index=False)

    output_path = tmp_path / "summary.csv"

    with pytest.raises(SystemExit) as exc:
        validate_main(
            [
                "--drivers",
                str(drivers_path),
                "--output",
                str(output_path),
            ]
        )

    assert exc.value.code == 1
    assert output_path.exists()

    summary = pd.read_csv(output_path)
    assert "ticker" in summary.columns
    assert (summary["status"] != "ok").any()

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "Validation flagged" in logged


def test_cli_passes_for_clean_dataset(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "period": [
                "2020-12-31",
                "2021-12-31",
                "2022-12-31",
                "2019-12-31",
                "2020-12-31",
                "2021-12-31",
            ],
        }
    )
    for feature in REQUIRED_FEATURES:
        if feature == "sales":
            df[feature] = [1e8, 1.1e8, 1.2e8, 2e8, 2.1e8, 2.2e8]
        else:
            df[feature] = [0.1, 0.12, 0.11, 0.2, 0.21, 0.19]

    drivers_path = tmp_path / "drivers.parquet"
    df.to_parquet(drivers_path, index=False)

    validate_main(["--drivers", str(drivers_path)])

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "Validation flagged" not in logged
