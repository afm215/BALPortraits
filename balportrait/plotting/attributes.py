"""Attribute association figures: Cramér's V matrix, phi panels and VIF bars."""

from __future__ import annotations

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from . import style

SERIF_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "serif"]}


def _label(name: str) -> str:
    return name.replace("_", " ").title()


def plot_cramers_v(matrix: pd.DataFrame):
    """Lower triangle of the corrected Cramér's V matrix."""
    style.apply_style()
    plt.rcParams.update(SERIF_RC)
    n = len(matrix)
    values = matrix.to_numpy(dtype=float).copy()
    values[np.triu(np.ones((n, n), dtype=bool), k=1)] = np.nan
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("white")
    norm = Normalize(0, 1)

    fig, ax = plt.subplots(figsize=(n * 0.7 + 2.5, n * 0.7 + 2.0))
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="equal")
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.03, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("Cramér's $V$")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([_label(c) for c in matrix.columns], rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels([_label(c) for c in matrix.index], fontsize=11)
    style.annotate_cells(ax, values, cmap, norm, fmt=".2f", fontsize=11)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=0.4)
        ax.axvline(k - 0.5, color="white", linewidth=0.4)
    ax.set_title("Attribute Association Matrix (Cramér's $V$)", pad=10)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Parameter")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def _orient(attr_a, attr_b, matrix):
    """Put gender on the y axis when it is paired with a hair attribute."""
    if "gender" in attr_b and "hair" in attr_a:
        return attr_b, attr_a, matrix.T
    return attr_a, attr_b, matrix


def plot_phi_panels(panels, max_units: int = 12, cell: float = 0.62):
    """Phi coefficient matrices of strongly associated attribute pairs, packed in rows.

    Parameters
    ----------
    panels
        List of ``(attr_a, attr_b, cramers_v, phi_matrix)``.
    """
    style.apply_style()
    plt.rcParams.update(SERIF_RC)
    panels = [(*_orient(a, b, m)[:2], v, _orient(a, b, m)[2]) for a, b, v, m in panels]
    rows, current, units = [], [], 0
    for idx, (_, _, _, matrix) in enumerate(panels):
        width = matrix.shape[1]
        if current and units + width > max_units:
            rows.append(current)
            current, units = [], 0
        current.append(idx)
        units += width
    if current:
        rows.append(current)

    heights = [max(panels[i][3].shape[0] for i in row) * cell + 2.2 for row in rows]
    fig_width = max(max(sum(panels[i][3].shape[1] * cell + 2.0 for i in row) for row in rows), 10)
    fig = plt.figure(figsize=(fig_width, max(sum(heights) * 1.15, 6)))
    outer = gridspec.GridSpec(len(rows), 1, height_ratios=heights, hspace=0.35, figure=fig)
    cmap = matplotlib.colormaps["RdBu"].copy()
    cmap.set_bad("white")
    norm = Normalize(-1, 1)
    for r, row in enumerate(rows):
        widths = [panels[i][3].shape[1] * cell + 2.0 for i in row]
        inner = gridspec.GridSpecFromSubplotSpec(1, len(row), subplot_spec=outer[r], width_ratios=widths, wspace=0.6)
        for c, idx in enumerate(row):
            attr_a, attr_b, v, matrix = panels[idx]
            values = matrix.to_numpy(dtype=float)
            ax = fig.add_subplot(inner[c])
            ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(range(values.shape[1]))
            ax.set_yticks(range(values.shape[0]))
            ax.set_xticklabels([_label(str(x)) for x in matrix.columns], rotation=20, ha="right", fontsize=9)
            ax.set_yticklabels([_label(str(y)) for y in matrix.index], fontsize=9)
            style.annotate_cells(ax, values, cmap, norm, fmt=".2f", fontsize=8)
            ax.set_title(f"{_label(attr_a)} $\\times$ {_label(attr_b)}\n$V={v:.2f}$", fontsize=10, pad=5)
            ax.set_xlabel(_label(attr_b), fontsize=9)
            ax.set_ylabel(_label(attr_a), fontsize=9)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
    fig.subplots_adjust(right=0.86)
    cbar_ax = fig.add_axes([0.90, 0.5, 0.012, 0.3])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, ticks=[-1, -0.5, 0, 0.5, 1]).set_label(
        r"$\phi$ (Pearson)", fontsize=9
    )
    return fig


DEMOGRAPHIC_COLOUR, STRUCTURAL_COLOUR, CONTROL_COLOUR = "#01696f", "#d19900", "#bab9b4"


def plot_vif(vif: pd.Series, top: int = 30):
    """Horizontal bars of the ``top`` highest VIF values of a design matrix."""
    style.apply_style()
    values = vif.sort_values(ascending=False).head(top)

    def colour(name):
        if name.startswith(("ethnicity", "gender", "age")):
            return DEMOGRAPHIC_COLOUR
        if "None" in name or "Bald" in name:
            return STRUCTURAL_COLOUR
        return CONTROL_COLOUR

    labels = [n.replace("hairstyle_texture_or_arrangement", "hairstyle").replace("_", " ").title() for n in values.index]
    fig, ax = plt.subplots(figsize=(10, 10))
    y = np.arange(len(values))
    ax.barh(y, values.to_numpy(), color=[colour(n) for n in values.index], height=0.72)
    ax.axvspan(1.0, 5.0, alpha=0.04, color="#437a22", zorder=0)
    ax.axvline(5.0, color="#a12c7b", linestyle="--", linewidth=1.1)
    for yi, value in zip(y, values.to_numpy()):
        if value >= 2.0:
            ax.text(value + 0.03, yi, f"{value:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.8)
    ax.invert_yaxis()
    ax.set_xlim(1.0, max(5.0, float(values.max()) + 0.3))
    ax.set_xlabel("VIF")
    ax.margins(y=0.01)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color="#e0ddd8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Patch(color=DEMOGRAPHIC_COLOUR, label="Demographic attr."),
            Patch(color=STRUCTURAL_COLOUR, label="Structural artefact"),
            Patch(color=CONTROL_COLOUR, label="Appearance control"),
        ],
        loc="lower right", framealpha=0.95,
    )
    return fig
