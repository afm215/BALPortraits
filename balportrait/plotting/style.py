"""Shared figure helpers: rc settings, saving, shading, annotations and palettes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_FORMATS = ("pdf", "png")

BASE_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
}


def apply_style() -> None:
    """Apply the base rc settings used by every figure of the package."""
    plt.rcParams.update(BASE_RC)


def save_figure(fig, path_stem, formats=DEFAULT_FORMATS) -> list[Path]:
    """Save ``fig`` as ``<path_stem>.<fmt>`` for every format, then close it."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = path_stem.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def shade_alternate(ax, n_groups: int, width: float = 1.0, offset: float = 0.0, start: int = 0, alpha=0.10):
    """Grey background band behind every other group (groups centred on ``offset + i * width``)."""
    for i in range(start, n_groups, 2):
        center = offset + i * width
        ax.axvspan(center - width / 2, center + width / 2, color="black", alpha=alpha, zorder=0, linewidth=0)


def text_color_for(rgba, threshold: float = 0.5) -> str:
    """Black or white text, whichever reads better on a cell of colour ``rgba``."""
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < threshold else "black"


def annotate_cells(ax, matrix, cmap, norm, fmt=".2f", fontsize=9, formatter=None, **text_kw):
    """Write the value of every non-NaN cell of ``matrix`` on an ``imshow`` axis."""
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            label = formatter(value) if formatter else f"{value:{fmt}}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=text_color_for(cmap(norm(value))),
                **text_kw,
            )


def tint_series(base_hex, n: int):
    """``n`` colours cycling through ``base_hex``, lightened with white on every new cycle."""
    bases = [mcolors.to_rgb(h) for h in base_hex]
    colours = []
    for i in range(n):
        base = bases[i % len(bases)]
        blend = min((i // len(bases)) * 0.30, 0.60)
        colours.append(tuple(b + (1.0 - b) * blend for b in base))
    return colours


# Okabe-Ito based families: ethnicity in blues/teal, gender in pink-purple, age in orange.
ETHNICITY_FAMILY = ("#0072B2", "#56B4E9", "#009E73")
GENDER_FAMILY = ("#CC79A7", "#E8A7C8")
AGE_FAMILY = ("#E69F00", "#D55E00", "#F0E442")
ATTRIBUTE_FAMILY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)
