import pandas as pd

from mlcoe_q1.pipelines import generate_cfo_recommendations


def test_generate_cfo_recommendations(tmp_path):
    forecaster = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "assets_mae": [1.0e10, 2.0e10],
            "equity_mae": [1.2e11, 5.0e10],
            "net_income_mae": [3.0e9, 8.0e9],
        }
    )
    forecaster_path = tmp_path / "forecaster.parquet"
    forecaster.to_parquet(forecaster_path)

    llm = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "mae": [1.1e11],
            "coverage": [0.4],
            "invalid_items": [2],
        }
    )
    llm_path = tmp_path / "llm.parquet"
    llm.to_parquet(llm_path)

    output_path = tmp_path / "recommendations.md"
    generate_cfo_recommendations.main(
        [
            "--forecaster-eval",
            str(forecaster_path),
            "--llm-eval",
            str(llm_path),
            "--output",
            str(output_path),
            "--top",
            "1",
        ]
    )

    content = output_path.read_text()
    assert "CFO Recommendations" in content
    assert "AAA" in content
    assert "Low coverage" in content

