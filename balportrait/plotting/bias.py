"""Rank-based figures: degree of bias, one-way mean ranks, intersectional matrices, score dispersion."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .. import config
from . import style

SIGNIFICANCE_COLORS = ("#00727a", "#3fb0c0", "#b3e2e9", "#ffffff")
RANK_CMAP = LinearSegmentedColormap.from_list("rank", ["#1a9850", "#ffffff", "#b2182b"])
RANK_HALF_RANGE = 2750 / 9000  # colour range of the matrices: N/2 ± 0.31 N (1,750–7,250 for N = 9,000)


def _grouped_bars(ax, table: pd.DataFrame, columns, models, width=0.16):
    """Draw one bar per column for every model (models on the x axis)."""
    x = np.arange(len(models))
    offsets = (np.arange(len(columns)) - (len(columns) - 1) / 2) * width
    for offset, column in zip(offsets, columns):
        ax.bar(
            x + offset,
            table.loc[models, column].to_numpy(dtype=float),
            width=width,
            color=config.SCENARIO_COLORS[column],
            label=config.SCENARIO_LABELS[column],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([config.display_name(m) for m in models], rotation=20, ha="right")
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    return x


def plot_degree_of_bias(dob: pd.DataFrame, significance: pd.DataFrame | None, equity_line: float | None, models):
    """Degree of bias per model and scenario, with the equity line and significance pies.

    Parameters
    ----------
    dob
        Output of :func:`balportrait.bias_metrics.dob_table`.
    significance
        Output of :func:`balportrait.significance.global_significance_summary` (pies),
        or None to omit them.
    equity_line
        DoB expected under demographic equity, or None to omit the line.
    """
    style.apply_style()
    models = [m for m in models if m in dob.index]
    columns = [c for c in ("global", *config.SCENARIOS) if dob[c].notna().any()]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(models) + 4), 4.2))
    x = _grouped_bars(ax, dob, columns, models)
    ax.set_ylabel("Degree of bias")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    if equity_line is not None:
        ax.set_xlim(-0.6, len(models) - 0.1)
        ax.axhline(equity_line, color="#444444", linestyle="--", linewidth=1.3, zorder=4)
        ax.text(
            0.997, equity_line, "Equity\nLine", transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7, fontweight="bold", color="#444444", linespacing=0.9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5}, zorder=5,
        )

    handles = [Patch(color=config.SCENARIO_COLORS[c], label=config.SCENARIO_LABELS[c]) for c in columns]
    scenario_legend = ax.legend(
        handles=handles, title="Scenarios", loc="upper left", bbox_to_anchor=(1.01, 0.62),
        frameon=True, edgecolor="#555555", title_fontproperties={"weight": "bold"},
    )

    if significance is not None:
        labels = [label for label, _, _ in config.P_VALUE_BINS]
        size = 0.55
        for xi, model in zip(x, models):
            if model not in significance.index:
                continue
            pie_ax = ax.inset_axes([xi - size / 2, 1.04, size, 0.22], transform=ax.get_xaxis_transform())
            fractions = significance.loc[model, labels].to_numpy(dtype=float)
            pie_ax.pie(
                fractions, colors=SIGNIFICANCE_COLORS, startangle=90, counterclock=False,
                wedgeprops={"edgecolor": "#bbbbbb", "linewidth": 0.6},
            )
            pie_ax.set_aspect("equal")
        ax.text(-0.01, 1.15, "Mann-Whitney U", transform=ax.transAxes, ha="right", va="center", fontsize=11)
        ax.add_artist(scenario_legend)
        sig_handles = [Patch(facecolor=c, edgecolor="#999999", label=l) for c, l in zip(SIGNIFICANCE_COLORS, labels)]
        ax.legend(
            handles=list(reversed(sig_handles)), title="Significativity", loc="lower left",
            bbox_to_anchor=(1.01, 0.64), frameon=True, edgecolor="#555555",
            title_fontproperties={"weight": "bold"},
        )
    return fig


ONE_WAY_GROUPS = (
    ("ethnicity", "E", config.ETHNICITIES, "o", ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")),
    ("age", "A", config.AGES, "^", ("#9467bd", "#8c564b", "#e377c2")),
    ("gender", "G", config.GENDERS, "s", ("#7f7f7f", "#bcbd22")),
)


def plot_one_way_ranks(mean_ranks: pd.DataFrame, models, scenarios=config.SCENARIOS, ncols: int = 2):
    """Mean portrait rank per ethnicity (E), age (A) and gender (G) group.

    Parameters
    ----------
    mean_ranks
        Output of :func:`balportrait.bias_metrics.mean_rank_table`.
    """
    style.apply_style()
    scenarios = [s for s in scenarios if s in set(mean_ranks["scenario"])]
    models = [m for m in models if m in set(mean_ranks["model"])]
    nrows = int(np.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(max(6.0, 1.45 * len(models)) * ncols, 3.4 * nrows), sharey=True, squeeze=False
    )
    lookup = mean_ranks.set_index(["model", "scenario", "attribute", "group"])["mean_rank"]
    block = 4.0  # x units per model: three sub-columns + gap
    for idx, scenario in enumerate(scenarios):
        ax = axes.flat[idx]
        for j, model in enumerate(models):
            base = j * block
            if j % 2 == 0:
                ax.axvspan(base - 0.6, base + 2.6, color="black", alpha=0.04, zorder=0, linewidth=0)
            for k, (attribute, _, groups, marker, colours) in enumerate(ONE_WAY_GROUPS):
                spread = np.linspace(-0.25, 0.25, len(groups))
                for offset, group, colour in zip(spread, groups, colours):
                    value = lookup.get((model, scenario, attribute, group), np.nan)
                    ax.scatter(base + k + offset, value, marker=marker, color=colour, s=24, zorder=3)
            for k in (0.5, 1.5):
                ax.axvline(base + k, color="#bbbbbb", linestyle=":", linewidth=0.8, zorder=1)
        ticks = [j * block + k for j in range(len(models)) for k in range(3)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([short for _ in models for _, short, *_ in ONE_WAY_GROUPS])
        ax.set_xlim(-0.6, (len(models) - 1) * block + 2.6)
        ax.set_title(config.SCENARIO_LABELS[scenario])
        if idx % ncols == 0:
            ax.set_ylabel("Average rank")
        if idx >= len(scenarios) - ncols:
            for j, model in enumerate(models):
                ax.text(
                    j * block + 1, -0.13, config.display_name(model), transform=ax.get_xaxis_transform(),
                    ha="right", va="top", rotation=30, fontsize=9,
                )
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)

    handles = [
        Line2D([], [], marker=marker, linestyle="", color=colour, label=f"{short}:{group}")
        for _, short, groups, marker, colours in ONE_WAY_GROUPS
        for group, colour in zip(groups, colours)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
    fig.tight_layout(h_pad=2.5)
    return fig


EGA_ROW_LABELS = {"African": "AF", "Asian": "AS", "Caucasian": "CA", "Indian": "IN"}
EGA_COLUMNS = [(g, a) for g in config.GENDERS for a in config.AGES]


def ega_matrix(mean_ranks: pd.DataFrame, model: str, scenario: str) -> np.ndarray:
    """4 × 6 matrix of EGA mean ranks: ethnicities × (gender, age)."""
    subset = mean_ranks[
        (mean_ranks["model"] == model) & (mean_ranks["scenario"] == scenario) & (mean_ranks["attribute"] == "ega")
    ].set_index("group")["mean_rank"]
    return np.array(
        [[subset.get(f"{e}|{g}|{a}", np.nan) for g, a in EGA_COLUMNS] for e in config.ETHNICITIES], dtype=float
    )


def plot_intersectional_matrices(mean_ranks: pd.DataFrame, models, scenarios=config.SCENARIOS, n_portraits=None):
    """Mean rank of every ethnicity × gender × age segment, one matrix per model and scenario.

    The colour scale is centred on the middle rank N/2.
    """
    style.apply_style()
    scenarios = [s for s in scenarios if s in set(mean_ranks["scenario"])]
    models = [m for m in models if m in set(mean_ranks["model"])]
    n_portraits = n_portraits or config.N_PORTRAITS
    center = n_portraits / 2
    norm = Normalize(vmin=center - RANK_HALF_RANGE * n_portraits, vmax=center + RANK_HALF_RANGE * n_portraits)

    fig, axes = plt.subplots(
        len(models), len(scenarios), figsize=(3.3 * len(scenarios) + 1.6, 2.35 * len(models) + 1.4), squeeze=False
    )
    image = None
    for i, model in enumerate(models):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]
            image = ax.imshow(ega_matrix(mean_ranks, model, scenario), cmap=RANK_CMAP, norm=norm, aspect="auto")
            ax.set_xticks(range(len(EGA_COLUMNS)))
            ax.set_yticks(range(len(config.ETHNICITIES)))
            ax.set_yticklabels([EGA_ROW_LABELS[e] for e in config.ETHNICITIES] if j == 0 else [], fontsize=12)
            if i == len(models) - 1:
                ax.set_xticklabels([f"{g[0]}-{a}" for g, a in EGA_COLUMNS], rotation=45, ha="right", fontsize=11)
            else:
                ax.set_xticklabels([])
            ax.tick_params(length=0)
            for k in range(1, len(EGA_COLUMNS)):
                ax.axvline(k - 0.5, color="#666666", linewidth=0.3)
            for k in range(1, len(config.ETHNICITIES)):
                ax.axhline(k - 0.5, color="#666666", linewidth=0.3)
            if i == 0:
                ax.set_title(config.SCENARIO_LABELS[scenario], fontsize=15)
        axes[i, 0].annotate(
            config.display_name(model), xy=(-0.32, 0.5), xycoords="axes fraction", ha="right", va="center",
            fontsize=15,
        )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.12, wspace=0.35, hspace=0.12)
    cbar_ax = fig.add_axes([0.55, 0.93, 0.4, 0.022])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="horizontal", ticks=[norm.vmin, center, norm.vmax])
    cbar.ax.set_xticklabels([f"{norm.vmin:.0f}", f"{center:.0f}", f"{norm.vmax:.0f}"])
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("Average rank")
    fig.text(
        0.04, 0.95, "Gender-age: F = Female · M = Male\nEthnicity: AF = African · AS = Asian · CA = Caucasian · IN = Indian",
        ha="left", va="center", fontsize=10,
    )
    return fig


def plot_score_dispersion(dispersion: pd.DataFrame, models):
    """Standard deviation of the portrait scores per model and scenario."""
    style.apply_style()
    models = [m for m in models if m in dispersion.index]
    columns = [c for c in config.SCENARIOS if dispersion[c].notna().any()]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.85 * len(models) + 3), 4.0))
    _grouped_bars(ax, dispersion, columns, models, width=0.2)
    style.shade_alternate(ax, len(models))
    ax.set_ylabel("Std of the portrait score")
    ax.legend(title="Scenarios", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    return fig
