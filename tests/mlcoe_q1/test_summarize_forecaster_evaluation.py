import pandas as pd

from mlcoe_q1.pipelines.summarize_forecaster_evaluation import (
    _coverage_rate,
    _parse_group_columns,
    _quantile_columns,
    summarize,
)


def test_parse_group_columns_handles_spaces():
    assert _parse_group_columns(" ticker , mode ,,") == ["ticker", "mode"]


def test_quantile_columns_parses_suffixes():
    columns = ["pred_total_assets_q10", "pred_total_assets_q90", "other"]
    mapping = _quantile_columns(columns, "pred_total_assets_q")
    assert mapping == {0.1: "pred_total_assets_q10", 0.9: "pred_total_assets_q90"}


def test_coverage_rate_computes_fraction():
    frame = pd.DataFrame(
        {
            "target": [1.0, 5.0, 10.0],
            "lower": [0.0, 6.0, 9.0],
            "upper": [2.0, 7.0, 11.0],
        }
    )
    coverage = _coverage_rate(frame, "target", "lower", "upper")
    assert coverage == 2 / 3


def test_summarize_produces_interval_metrics():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "mode": ["mlp", "mlp"],
            "assets_mae": [1.0, 3.0],
            "equity_mae": [2.0, 4.0],
            "identity_gap": [0.1, 0.3],
            "net_income_mae": [0.5, 0.7],
            "true_total_assets": [100.0, 120.0],
            "pred_total_assets_q10": [90.0, 110.0],
            "pred_total_assets_q90": [110.0, 130.0],
            "true_equity": [50.0, 60.0],
            "pred_equity_q10": [45.0, 55.0],
            "pred_equity_q90": [55.0, 65.0],
            "true_net_income": [5.0, 6.0],
            "pred_net_income_q10": [4.0, 5.0],
            "pred_net_income_q90": [6.0, 7.0],
        }
    )

    summary = summarize(df, ["ticker", "mode"])
    assert summary.loc[0, "observations"] == 2
    assert summary.loc[0, "assets_mae_mean"] == 2.0
    assert summary.loc[0, "assets_mae_median"] == 2.0
    assert summary.loc[0, "assets_mae_max"] == 3.0
    assert summary.loc[0, "equity_interval_width_mean"] == 10.0
    assert summary.loc[0, "assets_interval_coverage"] == 1.0

    summary_all = summarize(df, [])
    assert "ticker" not in summary_all.columns
    assert summary_all.loc[0, "assets_mae_mean"] == 2.0
