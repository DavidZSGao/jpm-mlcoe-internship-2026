import pandas as pd

from mlcoe_q1.pipelines import assess_scenario_reasonableness as scenarios


def build_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'ticker': 'AAA',
                'target_period': '2023-12-31',
                'horizon': 1,
                'mode': 'ensemble',
                'scenario': 'baseline',
                'pred_total_assets': 105.0,
                'pred_equity': 52.0,
                'pred_net_income': 11.0,
                'identity_gap': 0.2,
                'true_total_assets': 110.0,
                'true_equity': 55.0,
                'true_net_income': 10.0,
                'scenario_source_assets': 'point_estimate',
                'scenario_source_equity': 'point_estimate',
                'scenario_source_net_income': 'point_estimate',
            },
            {
                'ticker': 'AAA',
                'target_period': '2023-12-31',
                'horizon': 1,
                'mode': 'ensemble',
                'scenario': 'downside',
                'pred_total_assets': 90.0,
                'pred_equity': 45.0,
                'pred_net_income': 6.0,
                'identity_gap': -0.1,
                'true_total_assets': 110.0,
                'true_equity': 55.0,
                'true_net_income': 10.0,
                'scenario_source_assets': 'quantile',
                'scenario_source_equity': 'quantile',
                'scenario_source_net_income': 'quantile',
            },
            {
                'ticker': 'AAA',
                'target_period': '2023-12-31',
                'horizon': 1,
                'mode': 'ensemble',
                'scenario': 'upside',
                'pred_total_assets': 130.0,
                'pred_equity': 65.0,
                'pred_net_income': 14.0,
                'identity_gap': 0.05,
                'true_total_assets': 110.0,
                'true_equity': 55.0,
                'true_net_income': 10.0,
                'scenario_source_assets': 'quantile',
                'scenario_source_equity': 'quantile',
                'scenario_source_net_income': 'quantile',
            },
            {
                'ticker': 'BBB',
                'target_period': '2023-12-31',
                'horizon': 1,
                'mode': 'ensemble',
                'scenario': 'baseline',
                'pred_total_assets': 210.0,
                'pred_equity': 108.0,
                'pred_net_income': 21.0,
                'identity_gap': -0.02,
                'true_total_assets': 200.0,
                'true_equity': 100.0,
                'true_net_income': 20.0,
                'scenario_source_assets': 'point_estimate',
                'scenario_source_equity': 'point_estimate',
                'scenario_source_net_income': 'point_estimate',
            },
            {
                'ticker': 'BBB',
                'target_period': '2023-12-31',
                'horizon': 1,
                'mode': 'ensemble',
                'scenario': 'downside',
                'pred_total_assets': 180.0,
                'pred_equity': 95.0,
                'pred_net_income': 14.0,
                'identity_gap': -0.03,
                'true_total_assets': 200.0,
                'true_equity': 100.0,
                'true_net_income': 20.0,
                'scenario_source_assets': 'quantile',
                'scenario_source_equity': 'quantile',
                'scenario_source_net_income': 'quantile',
            },
            {
                'ticker': 'BBB',
                'target_period': '2023-12-31',
                'horizon': 1,
                'mode': 'ensemble',
                'scenario': 'upside',
                'pred_total_assets': 240.0,
                'pred_equity': 120.0,
                'pred_net_income': 18.0,
                'identity_gap': 0.04,
                'true_total_assets': 200.0,
                'true_equity': 100.0,
                'true_net_income': 20.0,
                'scenario_source_assets': 'quantile',
                'scenario_source_equity': 'quantile',
                'scenario_source_net_income': 'quantile',
            },
        ]
    )


def test_compute_scenario_statistics_returns_grouped_metrics():
    df = build_sample_dataframe()
    stats = scenarios.compute_scenario_statistics(df, ['scenario'])

    assert set(stats['scenario']) == {'baseline', 'downside', 'upside'}
    baseline = stats.loc[stats['scenario'] == 'baseline'].iloc[0]
    assert baseline['observations'] == 2
    assert baseline['total_assets_mae'] == 7.5  # mean(|105-110|, |210-200|)
    assert baseline['total_assets_bias'] == 2.5
    assert baseline['identity_gap_mae'] > 0
    assert baseline['quantile_rows'] == 0


def test_evaluate_scenarios_includes_interval_coverage():
    df = build_sample_dataframe()
    summary = scenarios.evaluate_scenarios(
        df,
        ['scenario'],
        lower='downside',
        upper='upside',
    )

    baseline = summary.loc[summary['scenario'] == 'baseline'].iloc[0]
    # Coverage metrics are merged back on the baseline row
    assert baseline['total_assets_interval_coverage'] == 1.0
    assert baseline['equity_interval_coverage'] == 1.0
    assert baseline['net_income_interval_coverage'] == 0.5
    # Width averages ((130-90) + (240-180)) / 2 = 50
    assert baseline['total_assets_interval_width'] == 50.0
    assert baseline['interval_observations'] == 2


def test_evaluate_scenarios_handles_overall_grouping():
    df = build_sample_dataframe()
    summary = scenarios.evaluate_scenarios(df, [], lower=None, upper=None)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row['observations'] == len(df)
    assert row['total_assets_mae'] is not None
