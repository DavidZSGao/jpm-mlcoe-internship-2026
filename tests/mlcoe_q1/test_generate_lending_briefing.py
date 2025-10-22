import pandas as pd

from mlcoe_q1.pipelines.generate_lending_briefing import build_briefing


def _sample_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["GM", "JPM"],
            "mode": ["mlp", "ensemble"],
            "observations": [6, 5],
            "assets_mae_mean": [1.2e9, 0.8e9],
            "equity_mae_mean": [0.4e9, 0.6e9],
            "net_income_mae_mean": [0.2e9, 0.35e9],
            "identity_gap_mean": [2.5e8, 7.5e8],
            "assets_interval_width_mean": [1.1e9, 1.5e9],
            "equity_interval_width_mean": [0.6e9, 0.9e9],
            "assets_interval_coverage": [0.94, 0.72],
            "equity_interval_coverage": [0.91, 0.68],
        }
    )


def test_build_briefing_flags_low_coverage_and_identity_gap():
    summary = _sample_summary()
    briefing = build_briefing(
        summary,
        group_cols=["ticker", "mode"],
        coverage_target=0.9,
        identity_threshold=5e8,
    )

    assert "# Strategic Lending Forecast Briefing" in briefing
    assert "GM - mlp" in briefing
    # JPM has both low coverage and a high identity gap, so it should be flagged.
    assert "⚠️" in briefing
    assert "Assets MAE $0.80B" in briefing
    assert "Coverage Assets 0.72" in briefing
    assert "Interval span Assets width $1.50B" in briefing


def test_build_briefing_table_contains_coverage_columns():
    summary = _sample_summary()
    briefing = build_briefing(
        summary,
        group_cols=["ticker", "mode"],
        coverage_target=0.8,
        identity_threshold=1e9,
    )

    assert "Assets Coverage" in briefing
    assert "Equity Coverage" in briefing


def test_build_briefing_requires_non_empty_summary():
    empty = pd.DataFrame()
    try:
        build_briefing(empty, group_cols=[], coverage_target=0.9, identity_threshold=1e9)
    except ValueError as exc:  # pragma: no cover - explicit assertion
        assert "Summary dataframe is empty" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for empty summary")

