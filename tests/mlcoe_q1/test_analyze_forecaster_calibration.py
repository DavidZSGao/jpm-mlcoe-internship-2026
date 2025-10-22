import numpy as np
import pandas as pd

from mlcoe_q1.pipelines import analyze_forecaster_calibration


def test_compute_calibration_grouped():
    df = pd.DataFrame(
        {
            "ticker": ["GM", "GM", "F", "F"],
            "mode": ["mlp", "mlp", "mlp", "mlp"],
            "true_total_assets": [100.0, 120.0, 90.0, 95.0],
            "pred_total_assets_q10": [80.0, 100.0, 70.0, 85.0],
            "pred_total_assets_q90": [130.0, 140.0, 105.0, 120.0],
            "true_equity": [40.0, 48.0, 30.0, 32.0],
            "pred_equity_q10": [30.0, 35.0, 20.0, 25.0],
            "pred_equity_q90": [55.0, 60.0, 40.0, 45.0],
        }
    )

    grouped = analyze_forecaster_calibration.compute_calibration(
        df,
        group_cols=["ticker"],
        metrics=["assets", "equity"],
    )

    assert set(grouped["ticker"]) == {"GM", "F"}
    gm_row = grouped.set_index("ticker").loc["GM"]
    assert np.isclose(gm_row["assets_q10_coverage"], 0.0)
    assert np.isclose(gm_row["assets_q10_error"], -0.1)
    assert np.isclose(gm_row["assets_q90_coverage"], 1.0)


def test_compute_calibration_handles_missing_quantiles():
    df = pd.DataFrame(
        {
            "ticker": ["GM"],
            "mode": ["mlp"],
            "true_total_assets": [100.0],
            "pred_total_assets": [102.0],
        }
    )

    result = analyze_forecaster_calibration.compute_calibration(
        df,
        group_cols=["ticker"],
        metrics=["assets"],
    )

    assert result.empty


def test_parse_tokens():
    tokens = analyze_forecaster_calibration._parse_tokens("ticker, mode , horizon")
    assert tokens == ["ticker", "mode", "horizon"]

    assert analyze_forecaster_calibration._parse_tokens("  ") == []
