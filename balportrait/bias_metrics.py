"""Rank-based bias metrics: group mean ranks, degree of bias (DoB) and its null baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, io, scoring


def group_mean_ranks(scores: pd.DataFrame, rank_col: str, by: str) -> pd.Series:
    """Mean rank of every group of ``by`` (``"ethnicity"``, ``"gender"``, ``"age"`` or ``"ega"``)."""
    return scores.dropna(subset=[rank_col]).groupby(by)[rank_col].mean()


def degree_of_bias(scores: pd.DataFrame, rank_col: str, by: str = "ega") -> float:
    """Standard deviation of the group mean ranks (24 EGA segments by default)."""
    return float(group_mean_ranks(scores, rank_col, by).std(ddof=1))


def dob_table(scores_by_model: dict) -> pd.DataFrame:
    """Degree of bias per model and scenario.

    The ``global`` column is the mean of the four scenario DoBs (NaN when a model
    lacks a scenario).
    """
    rows = {}
    for model, scores in scores_by_model.items():
        rows[model] = {s: degree_of_bias(scores, f"rank_{s}") for s in scoring.available_scenarios(scores)}
    table = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(config.SCENARIOS))
    table.insert(0, "global", table[list(config.SCENARIOS)].mean(axis=1, skipna=False))
    table.index.name = "model"
    return table


def null_baseline(metadata: pd.DataFrame, n_seeds: int = 20, seed: int = 0, scenarios=config.SCENARIOS):
    """DoB expected under demographic equity.

    Every portrait-item fitness score is replaced by an i.i.d. random integer in
    [1, 10]; item lists, prestige weights, aggregation and ranking are unchanged.

    Returns
    -------
    pd.DataFrame
        Columns ``seed``, ``scenario``, ``dob``; one row per seed and scenario.
    """
    rng = np.random.default_rng(seed)
    records = []
    for run in range(n_seeds):
        for scenario in scenarios:
            prestige = io.load_prestige(scenario)
            fitness = scoring.random_fitness(metadata.index, prestige.index, rng)
            ranks = scoring.scores_from_fitness(fitness, prestige)[["rank"]].join(metadata)
            records.append({"seed": run, "scenario": scenario, "dob": degree_of_bias(ranks, "rank")})
    return pd.DataFrame(records)


def expected_null_dob(n_portraits: int = config.N_PORTRAITS, n_groups: int = 24) -> float:
    """Closed-form std of a group mean rank when ranks are exchangeable across groups."""
    group_size = n_portraits / n_groups
    rank_std = np.sqrt((n_portraits**2 - 1) / 12)
    finite_population = np.sqrt((n_portraits - group_size) / (n_portraits - 1))
    return float(rank_std / np.sqrt(group_size) * finite_population)


def mean_rank_table(scores_by_model: dict, by=("ethnicity", "age", "gender")) -> pd.DataFrame:
    """Long table of group mean ranks: ``model, scenario, attribute, group, mean_rank, n``."""
    records = []
    for model, scores in scores_by_model.items():
        for scenario in scoring.available_scenarios(scores):
            rank_col = f"rank_{scenario}"
            valid = scores.dropna(subset=[rank_col])
            for attribute in by:
                stats = valid.groupby(attribute)[rank_col].agg(["mean", "size"])
                for group, row in stats.iterrows():
                    records.append(
                        {
                            "model": model,
                            "scenario": scenario,
                            "attribute": attribute,
                            "group": group,
                            "mean_rank": row["mean"],
                            "n": int(row["size"]),
                        }
                    )
    return pd.DataFrame(records)


def score_dispersion_table(scores_by_model: dict) -> pd.DataFrame:
    """Standard deviation of the portrait scores per model and scenario."""
    rows = {
        model: {s: scores[f"score_{s}"].std() for s in scoring.available_scenarios(scores)}
        for model, scores in scores_by_model.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=list(config.SCENARIOS))
    table.index.name = "model"
    return table
