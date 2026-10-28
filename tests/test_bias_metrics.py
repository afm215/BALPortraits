import numpy as np
import pandas as pd

from balportrait import bias_metrics, config, io


def test_degree_of_bias_is_std_of_segment_means(toy_ranks):
    means = toy_ranks.groupby("ega")["rank"].mean()
    assert np.isclose(bias_metrics.degree_of_bias(toy_ranks, "rank"), means.std(ddof=1))


def test_degree_of_bias_is_zero_when_segments_are_equal(toy_ranks):
    equal = toy_ranks.assign(rank=5.0)
    assert bias_metrics.degree_of_bias(equal, "rank") == 0


def test_global_dob_is_mean_of_scenario_dobs(toy_ranks, rng):
    scores = toy_ranks.drop(columns="rank")
    for scenario in config.SCENARIOS:
        scores[f"rank_{scenario}"] = rng.permutation(len(scores)) + 1.0
    table = bias_metrics.dob_table({"toy": scores})
    assert np.isclose(table.loc["toy", "global"], table.loc["toy", list(config.SCENARIOS)].mean())


def test_null_baseline_matches_closed_form():
    null = bias_metrics.null_baseline(io.load_metadata(), n_seeds=3, scenarios=("brands",))
    expected = bias_metrics.expected_null_dob()
    assert 125 < expected < 135
    assert (null["dob"].between(0.5 * expected, 1.5 * expected)).all()


def test_mean_rank_table_shape(toy_ranks):
    scores = toy_ranks.rename(columns={"rank": "rank_brands"}).assign(score_brands=0.0)
    table = bias_metrics.mean_rank_table({"toy": scores}, by=("ethnicity", "ega"))
    assert set(table["attribute"]) == {"ethnicity", "ega"}
    assert len(table) == 4 + 24
    assert isinstance(table, pd.DataFrame)
