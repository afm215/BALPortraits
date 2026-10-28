import numpy as np
import pandas as pd
import pytest

from balportrait import scoring


def test_portrait_score_follows_equation_1():
    fitness = pd.DataFrame({1: [1.0, 5.0, 10.0], 2: [10.0, 4.0, 1.0]}, index=["a", "b", "c"])
    prestige = pd.Series({1: 80.0, 2: 20.0})
    weights = (prestige - prestige.mean()) / prestige.std()
    normalized = (fitness - fitness.min()) / (fitness.max() - fitness.min())
    expected = (normalized * weights.to_numpy()).sum(axis=1) / 2
    pd.testing.assert_series_equal(scoring.portrait_scores(fitness, prestige), expected)


def test_constant_item_is_ignored():
    fitness = pd.DataFrame({1: [1.0, 5.0], 2: [3.0, 3.0]})
    with pytest.warns(UserWarning):
        normalized = scoring.minmax_per_item(fitness)
    assert normalized[2].isna().all()
    with pytest.warns(UserWarning):
        scores = scoring.portrait_scores(fitness, pd.Series({1: 2.0, 2: 1.0}))
    assert np.isfinite(scores).all()


def test_rank_one_is_the_highest_score_and_ties_share_the_lowest_rank():
    ranks = scoring.rank_scores(pd.Series([0.1, 0.5, 0.5, -0.2]))
    assert ranks.tolist() == [3.0, 1.0, 1.0, 4.0]


def test_missing_scores_are_skipped():
    fitness = pd.DataFrame({1: [1.0, 10.0, np.nan], 2: [2.0, 8.0, 5.0]})
    scores = scoring.portrait_scores(fitness, pd.Series({1: 1.0, 2: 3.0}))
    assert scores.notna().all()
