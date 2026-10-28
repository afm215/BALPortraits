import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from balportrait import config, significance


def test_rank_biserial_sign_convention():
    scores = pd.DataFrame({"rank": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "group": ["A", "A", "A", "B", "B", "B"]})
    table = significance.pairwise_mwu(scores, "rank", "group")
    assert table.loc[0, "r_rb"] == 1.0  # A always ranked better than B
    assert table.loc[0, "delta_mean_rank"] == -3.0


def test_ega_pairs(toy_ranks):
    table = significance.pairwise_mwu(toy_ranks, "rank", "ega")
    assert len(table) == 276
    assert table["r_rb"].between(-1, 1).all()


def test_benjamini_hochberg():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    assert np.allclose(significance.adjust_pvalues(p, "fdr_bh"), multipletests(p, method="fdr_bh")[1])
    assert np.array_equal(significance.adjust_pvalues(p, "none"), p)


def test_global_significance_bins_sum_to_one(toy_ranks, rng):
    scores = toy_ranks.rename(columns={"rank": "global_rank"})
    summary = significance.global_significance_summary({"toy": scores})
    labels = [label for label, _, _ in config.P_VALUE_BINS]
    assert np.isclose(summary.loc["toy", labels].sum(), 1.0)
    assert summary.loc["toy", "n_tests"] == 276
