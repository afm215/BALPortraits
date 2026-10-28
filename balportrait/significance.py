"""Pairwise Mann-Whitney U tests and rank-biserial correlations between demographic groups."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

from . import config, scoring


def rank_biserial(u_a: float, n_a: int, n_b: int) -> float:
    """Rank-biserial correlation ``1 - 2 U_A / (n_A n_B)``.

    ``U_A`` is the Mann-Whitney statistic of group A on portrait ranks. Since rank 1
    is the best position, a positive value means that group A is ranked better than
    group B.
    """
    return 1.0 - 2.0 * u_a / (n_a * n_b)


def adjust_pvalues(p_values, method: str = "none") -> np.ndarray:
    """Adjust p-values for multiple comparisons (``"none"`` or a statsmodels method, e.g. ``"fdr_bh"``)."""
    p_values = np.asarray(p_values, dtype=float)
    if method == "none" or len(p_values) == 0:
        return p_values
    return multipletests(p_values, method=method)[1]


def pairwise_mwu(scores: pd.DataFrame, rank_col: str, by: str, alternative: str = "two-sided", order=None):
    """Two-sided Mann-Whitney U test for every pair of groups of ``by``.

    Parameters
    ----------
    order
        Group order defining which group is A in each pair (default: sorted labels).

    Returns
    -------
    pd.DataFrame
        ``group_a, group_b, n_a, n_b, mean_rank_a, mean_rank_b, delta_mean_rank, U, p_value, r_rb``.
    """
    valid = scores.dropna(subset=[rank_col, by])
    groups = {name: g[rank_col].to_numpy() for name, g in valid.groupby(by)}
    labels = [g for g in (order or sorted(groups)) if g in groups]
    records = []
    for a, b in combinations(labels, 2):
        ranks_a, ranks_b = groups[a], groups[b]
        u_a, p_value = mannwhitneyu(ranks_a, ranks_b, alternative=alternative)
        records.append(
            {
                "group_a": a,
                "group_b": b,
                "n_a": len(ranks_a),
                "n_b": len(ranks_b),
                "mean_rank_a": ranks_a.mean(),
                "mean_rank_b": ranks_b.mean(),
                "delta_mean_rank": ranks_a.mean() - ranks_b.mean(),
                "U": u_a,
                "p_value": p_value,
                "r_rb": rank_biserial(u_a, len(ranks_a), len(ranks_b)),
            }
        )
    return pd.DataFrame(records)


def pairwise_tables(
    scores_by_model: dict,
    attributes=("gender", "age", "ethnicity", "ega"),
    p_adjust: str = "none",
) -> pd.DataFrame:
    """Pairwise tests for every model, scenario and attribute, as one long table.

    When ``p_adjust`` is not ``"none"``, p-values are adjusted within each
    model × scenario × attribute family (column ``p_adj``).
    """
    tables = []
    for model, scores in scores_by_model.items():
        for scenario in scoring.available_scenarios(scores):
            for attribute in attributes:
                table = pairwise_mwu(scores, f"rank_{scenario}", attribute)
                table["p_adj"] = adjust_pvalues(table["p_value"], p_adjust)
                table.insert(0, "attribute", attribute)
                table.insert(0, "scenario", scenario)
                table.insert(0, "model", model)
                tables.append(table)
    return pd.concat(tables, ignore_index=True)


def global_significance_summary(scores_by_model: dict, p_adjust: str = "fdr_bh") -> pd.DataFrame:
    """Share of EGA pairs per significance bin, on the global rank of each model.

    The global rank of a portrait is the mean of its four scenario ranks. All
    pairwise tests between the 24 EGA segments (276 pairs) are run and adjusted
    within the model.
    """
    records = []
    for model, scores in scores_by_model.items():
        if "global_rank" not in scores.columns:
            continue
        table = pairwise_mwu(scores, "global_rank", "ega")
        p_adj = adjust_pvalues(table["p_value"], p_adjust)
        record = {"model": model, "n_tests": len(table)}
        for label, low, high in config.P_VALUE_BINS:
            record[label] = float(np.mean((p_adj >= low) & (p_adj < high)))
        records.append(record)
    return pd.DataFrame(records).set_index("model")


def mean_abs_rrb_table(pairwise_long: pd.DataFrame, attribute: str = "ega") -> pd.DataFrame:
    """Mean |r_rb| over all pairs of ``attribute``, per model (rows) and scenario (columns)."""
    subset = pairwise_long[pairwise_long["attribute"] == attribute]
    table = subset.assign(abs_r_rb=subset["r_rb"].abs()).pivot_table(
        index="model", columns="scenario", values="abs_r_rb", aggfunc="mean"
    )
    table = table.reindex(index=config.order_models(table.index), columns=[s for s in config.SCENARIOS if s in table])
    table["average_across_scenarios"] = table.mean(axis=1)
    return table


def ranking_agreement(mean_rrb: pd.DataFrame, dob: pd.DataFrame):
    """Compare, per scenario, the ordering of models by mean |r_rb| and by degree of bias.

    Returns
    -------
    comparison : pd.DataFrame
        ``scenario, model, mean_abs_rrb, dob, rank_by_rrb, rank_by_dob, rank_diff``.
    agreement : pd.DataFrame
        ``scenario, n_models, spearman_rho, spearman_p, kendall_tau, kendall_p``.
    """
    comparison, agreement = [], []
    for scenario in [s for s in config.SCENARIOS if s in mean_rrb.columns and s in dob.columns]:
        both = pd.concat([mean_rrb[scenario].rename("mean_abs_rrb"), dob[scenario].rename("dob")], axis=1).dropna()
        both["rank_by_rrb"] = both["mean_abs_rrb"].rank(ascending=False, method="min").astype(int)
        both["rank_by_dob"] = both["dob"].rank(ascending=False, method="min").astype(int)
        both["rank_diff"] = both["rank_by_rrb"] - both["rank_by_dob"]
        comparison.append(both.rename_axis("model").reset_index().assign(scenario=scenario))
        if len(both) >= 2:
            rho, rho_p = spearmanr(both["rank_by_rrb"], both["rank_by_dob"])
            tau, tau_p = kendalltau(both["rank_by_rrb"], both["rank_by_dob"])
        else:
            rho = rho_p = tau = tau_p = np.nan
        agreement.append(
            {
                "scenario": scenario,
                "n_models": len(both),
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "kendall_tau": tau,
                "kendall_p": tau_p,
            }
        )
    columns = ["scenario", "model", "mean_abs_rrb", "dob", "rank_by_rrb", "rank_by_dob", "rank_diff"]
    comparison = pd.concat(comparison, ignore_index=True)[columns].sort_values(["scenario", "rank_by_rrb"])
    return comparison, pd.DataFrame(agreement)
