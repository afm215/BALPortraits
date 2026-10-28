"""Heatmaps of pairwise Mann-Whitney U p-values and rank-biserial correlations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from .. import config
from . import style

_PVAL_COLOURS = ["#a50026", "#d73027", "#f46d43", "#fdae61", "#a6d96a", "#1a9850"]
CMAP_PVALUE = LinearSegmentedColormap.from_list("pval", _PVAL_COLOURS)
CMAP_RRB = LinearSegmentedColormap.from_list("rrb", _PVAL_COLOURS[::-1])
RRB_VMAX = 0.55

ONE_WAY_ORDER = {
    "gender": list(config.GENDERS),
    "age": list(config.AGES),
    "ethnicity": list(config.ETHNICITIES),
}
ONE_WAY_TITLES = {"gender": "Gender", "age": "Age", "ethnicity": "Ethnicity"}

# EGA segments ordered by gender, then age, then ethnicity (block structure of the joint heatmaps).
EGA_ORDER = [f"{e}|{g}|{a}" for g in config.GENDERS for a in config.AGES for e in config.ETHNICITIES]


def _value_settings(value: str):
    if value == "p_value":
        return "p_value", CMAP_PVALUE, Normalize(0, 1), "p-value"
    if value == "r_rb":
        return "abs_r_rb", CMAP_RRB, Normalize(0, RRB_VMAX), "Rank-biserial |r|"
    raise ValueError("value must be 'p_value' or 'r_rb'")


def pair_matrix(pairs: pd.DataFrame, order, column: str) -> np.ndarray:
    """Symmetric matrix of ``column`` over the groups of ``order`` (NaN on the diagonal)."""
    index = {g: i for i, g in enumerate(order)}
    matrix = np.full((len(order), len(order)), np.nan)
    for _, row in pairs.iterrows():
        if row["group_a"] in index and row["group_b"] in index:
            i, j = index[row["group_a"]], index[row["group_b"]]
            matrix[i, j] = matrix[j, i] = row[column]
    return matrix


def _format(value: float, is_pvalue: bool, digits: int) -> str:
    return "<.001" if is_pvalue and value < 0.001 else f"{value:.{digits}f}"


def _subset(pairwise_long, model, scenario, attribute):
    mask = (
        (pairwise_long["model"] == model)
        & (pairwise_long["scenario"] == scenario)
        & (pairwise_long["attribute"] == attribute)
    )
    subset = pairwise_long[mask]
    return subset.assign(abs_r_rb=subset["r_rb"].abs())


def plot_pairwise_heatmaps(pairwise_long: pd.DataFrame, models, value="p_value", scenarios=config.SCENARIOS,
                           attributes=("gender", "age", "ethnicity")):
    """One matrix per scenario (rows), model and one-way attribute (columns)."""
    style.apply_style()
    column, cmap, norm, label = _value_settings(value)
    scenarios = [s for s in scenarios if s in set(pairwise_long["scenario"])]
    n_cols = len(models) * len(attributes)
    fig, axes = plt.subplots(
        len(scenarios), n_cols, figsize=(2.6 * n_cols + 1.5, 2.7 * len(scenarios) + 0.8), squeeze=False
    )
    for r, scenario in enumerate(scenarios):
        for m, model in enumerate(models):
            for a, attribute in enumerate(attributes):
                ax = axes[r, m * len(attributes) + a]
                order = ONE_WAY_ORDER[attribute]
                matrix = pair_matrix(_subset(pairwise_long, model, scenario, attribute), order, column)
                ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
                style.annotate_cells(
                    ax, matrix, cmap, norm, fontsize=9, fontweight="bold",
                    formatter=lambda v: _format(v, value == "p_value", 3),
                )
                ax.set_xticks(range(len(order)))
                ax.set_yticks(range(len(order)))
                ax.set_xticklabels(order, fontsize=9, rotation=30 if len(order) > 3 else 0, ha="right")
                ax.set_yticklabels(order, fontsize=9, rotation=70 if len(order) > 3 else 0, va="center")
                ax.set_title(ONE_WAY_TITLES[attribute], fontsize=10, fontweight="bold", pad=4)
                for spine in ax.spines.values():
                    spine.set_visible(False)
        axes[r, 0].annotate(
            scenario, xy=(-0.55, 0.5), xycoords="axes fraction", rotation=90, ha="center", va="center",
            fontsize=11, fontweight="bold",
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 0.95))
    for m, model in enumerate(models):
        first = axes[0, m * len(attributes)].get_position()
        last = axes[0, (m + 1) * len(attributes) - 1].get_position()
        fig.text((first.x0 + last.x1) / 2, 0.965, config.display_name(model), ha="center", fontsize=12,
                 fontweight="bold")
        if m < len(models) - 1:
            x_sep = (last.x1 + axes[0, (m + 1) * len(attributes)].get_position().x0) / 2
            fig.add_artist(Line2D([x_sep, x_sep], [0.02, 0.96], transform=fig.transFigure, color="#bbbbbb",
                                  linewidth=1.5, linestyle="--"))
    cbar_ax = fig.add_axes([0.955, 0.15, 0.012, 0.65])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax).set_label(label)
    return fig


def _short_ega(label: str) -> str:
    ethnicity, gender, age = label.split("|")
    return f"{gender[:3]}|{age}|{ethnicity[:3]}"


def plot_joint_heatmaps(pairwise_long: pd.DataFrame, models, value="p_value", scenarios=config.SCENARIOS):
    """24 × 24 EGA matrices, one per scenario (rows) and model (columns)."""
    style.apply_style()
    column, cmap, norm, label = _value_settings(value)
    scenarios = [s for s in scenarios if s in set(pairwise_long["scenario"])]
    n = len(EGA_ORDER)
    tile = 0.68
    fig, axes = plt.subplots(
        len(scenarios), len(models), figsize=(n * tile * len(models) + 4, n * tile * len(scenarios) + 3),
        squeeze=False,
    )
    block, sub_block = len(config.AGES) * len(config.ETHNICITIES), len(config.ETHNICITIES)
    short = [_short_ega(g) for g in EGA_ORDER]
    for r, scenario in enumerate(scenarios):
        for c, model in enumerate(models):
            ax = axes[r, c]
            matrix = pair_matrix(_subset(pairwise_long, model, scenario, "ega"), EGA_ORDER, column)
            ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
            style.annotate_cells(
                ax, matrix, cmap, norm, fontsize=12, fontweight="bold",
                formatter=lambda v: _format(v, value == "p_value", 2),
            )
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(short if r == len(scenarios) - 1 else [], rotation=90, fontsize=14,
                               fontfamily="monospace")
            ax.set_yticklabels(short if c == 0 else [], fontsize=14, fontfamily="monospace")
            for sep in range(sub_block, n, sub_block):
                width, alpha = (2.0, 1.0) if sep % block == 0 else (0.8, 0.5)
                ax.axhline(sep - 0.5, color="white", linewidth=width, alpha=alpha)
                ax.axvline(sep - 0.5, color="white", linewidth=width, alpha=alpha)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(config.display_name(model), fontsize=19, fontweight="bold", pad=6)
        axes[r, 0].annotate(
            scenario, xy=(-0.2, 0.5), xycoords="axes fraction", rotation=90, ha="center", va="center",
            fontsize=19, fontweight="bold",
        )
    fig.subplots_adjust(left=0.1, right=0.91, bottom=0.07, top=0.97, wspace=0.08, hspace=0.04)
    cbar_ax = fig.add_axes([0.935, 0.3, 0.012, 0.4])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax).set_label(
        "p-value" if value == "p_value" else "|r_rb|", fontsize=17
    )
    return fig
