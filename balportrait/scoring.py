"""Portrait prestige scores (Eq. 1 of the paper) and portrait ranks."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, io


def standardize_prestige(prestige: pd.Series) -> pd.Series:
    """Standardize item prestige within a scenario: ``(p - mean) / std``."""
    return (prestige - prestige.mean()) / prestige.std()


def minmax_per_item(fitness: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize every item column over the portraits.

    An item with a constant score carries no information and is set to NaN, so that
    it is ignored in the portrait mean.
    """
    low, high = fitness.min(), fitness.max()
    span = high - low
    constant = span == 0
    if constant.any():
        warnings.warn(f"Items with a constant score are ignored: {list(span.index[constant])}")
    return (fitness - low) / span.where(~constant)


def portrait_scores(fitness: pd.DataFrame, prestige: pd.Series) -> pd.Series:
    """Portrait prestige score ``a_S = mean_i( p̃_i · f̂_i )``.

    Parameters
    ----------
    fitness
        Raw fitness scores, one row per portrait and one column per item id.
    prestige
        Raw prestige of the items, indexed by item id.
    """
    weights = standardize_prestige(prestige).loc[fitness.columns]
    normalized = minmax_per_item(fitness)
    return (normalized * weights.to_numpy()).mean(axis=1, skipna=True)


def rank_scores(scores: pd.Series) -> pd.Series:
    """Rank portraits by score, rank 1 being the highest score (ties share the lowest rank)."""
    return scores.rank(ascending=False, method="min")


def scores_from_fitness(fitness: pd.DataFrame, prestige: pd.Series) -> pd.DataFrame:
    """Score and rank columns for one scenario, indexed like ``fitness``."""
    score = portrait_scores(fitness, prestige)
    return pd.DataFrame({"score": score, "rank": rank_scores(score)})


def compute_model_scores(
    outputs_dir: Path,
    model: str,
    metadata: pd.DataFrame,
    scenarios=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Scores and ranks of one model for every available scenario.

    Returns
    -------
    pd.DataFrame
        One row per dataset portrait (``metadata`` index), with ``score_<scenario>``
        and ``rank_<scenario>`` columns, ``global_rank`` (mean of the four scenario
        ranks, NaN unless the portrait has all four) and the metadata columns.
        Portraits without output for a scenario have NaN score and rank there.
    """
    available = io.discover_models(outputs_dir).get(model, [])
    scenarios = [s for s in (scenarios or config.SCENARIOS) if s in available]
    if not scenarios:
        raise FileNotFoundError(f"No output found for model '{model}' in {outputs_dir}")

    columns = {}
    for scenario in scenarios:
        fitness = io.load_model_output(outputs_dir, scenario, model, metadata.index, verbose=verbose)
        result = scores_from_fitness(fitness, io.load_prestige(scenario))
        columns[f"score_{scenario}"] = result["score"]
        columns[f"rank_{scenario}"] = result["rank"]
    scores = pd.DataFrame(columns).reindex(metadata.index)
    if set(config.SCENARIOS) <= set(scenarios):
        scores["global_rank"] = scores[[f"rank_{s}" for s in config.SCENARIOS]].mean(axis=1, skipna=False)
    return scores.join(metadata)


def compute_all_scores(outputs_dir: Path, models, metadata: pd.DataFrame, verbose: bool = True) -> dict:
    """``{model: compute_model_scores(...)}`` for every model, in paper order."""
    return {
        model: compute_model_scores(outputs_dir, model, metadata, verbose=verbose)
        for model in config.order_models(models)
    }


def available_scenarios(scores: pd.DataFrame) -> list[str]:
    """Scenarios that have a rank column in a score table."""
    return [s for s in config.SCENARIOS if f"rank_{s}" in scores.columns]


def random_fitness(index: pd.Index, items: pd.Index, rng: np.random.Generator, low: int = 1, high: int = 10):
    """I.i.d. uniform integer fitness scores in ``[low, high]`` (null baseline)."""
    values = rng.integers(low, high + 1, size=(len(index), len(items)))
    return pd.DataFrame(values.astype(float), index=index, columns=items)
