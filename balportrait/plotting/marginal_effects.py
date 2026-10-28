"""Marginal-effect bar charts and regression residual diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from .. import config
from . import style

ALPHA_SIGNIFICANT = 0.85
ALPHA_NON_SIGNIFICANT = 0.25

# Paul Tol's muted then bright qualitative palettes (attribute figures).
ATTRIBUTE_COLOURS = (
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499",
    "#DDDDDD", "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB",
)


def level_label(attribute: str, level: str) -> str:
    """Display label of an attribute level (booleans as Present / Absent)."""
    if level == "True" or (attribute == "moustache_style" and level != "None"):
        return "Present"
    if level == "False":
        return "Absent"
    return level.replace("_", " ").title()


def attribute_label(attribute: str) -> str:
    return "Moustache" if attribute == "moustache_style" else attribute.replace("_", " ").title()


def _levels(coefs: pd.DataFrame, attribute: str, preferred=None):
    present = list(dict.fromkeys(coefs.loc[coefs["attribute"] == attribute, "level"]))
    if preferred:
        ordered = [lvl for lvl in preferred if lvl in present]
        return ordered + [lvl for lvl in present if lvl not in ordered]
    return present


def _draw_effect_bars(ax, coefs: pd.DataFrame, models, slots, colours, alpha=config.ME_ALPHA):
    """Bars of every (attribute, level) slot for every model; ``None`` slots leave a gap.

    Bars are opaque when ``p_adj`` ≤ alpha and transparent otherwise; error bars are
    the half-width of the confidence interval.
    """
    lookup = coefs.set_index(["model", "attribute", "level"])
    width = 0.8 / len(slots)
    for i, model in enumerate(models):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="black", alpha=0.10, zorder=0, linewidth=0)
        for k, slot in enumerate(slots):
            if slot is None or (model, *slot) not in lookup.index:
                continue
            row = lookup.loc[(model, *slot)]
            x = i - 0.4 + (k + 0.5) * width
            opacity = ALPHA_SIGNIFICANT if row["p_adj"] <= alpha else ALPHA_NON_SIGNIFICANT
            ax.bar(x, row["estimate"], width=width, color=colours[slot], alpha=opacity, zorder=3)
            ax.errorbar(
                x, row["estimate"], yerr=row["half_width"], fmt="none", color="black", linewidth=1.0,
                capsize=3, capthick=1.0, alpha=opacity, zorder=5,
            )
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_xticks(range(len(models)))


def _panel_grid(n_panels: int, n_models: int, ncols: int):
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(max(n_models * 1.4, 7) * ncols, 5.0 * nrows), squeeze=False, sharex=True
    )
    return fig, axes, ncols


def _finish_panels(fig, axes, scenarios, models, ncols, ylabel="Marginal effect"):
    for idx, scenario in enumerate(scenarios):
        ax = axes.flat[idx]
        ax.set_title(config.SCENARIO_LABELS[scenario], fontsize=20, pad=8)
        ax.tick_params(axis="y", labelsize=14)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=16)
        if idx >= len(scenarios) - ncols:
            ax.set_xticklabels([config.display_name(m) for m in models], fontsize=14)
        else:
            ax.tick_params(labelbottom=False)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    fig.tight_layout()


def plot_demographic_effects(coefs: pd.DataFrame, models, scenarios=config.SCENARIOS, ncols: int = 2,
                             alpha=config.ME_ALPHA):
    """Ethnicity, gender and age marginal effects per model, one panel per scenario."""
    style.apply_style()
    scenarios = [s for s in scenarios if s in set(coefs["scenario"])]
    models = [m for m in models if m in set(coefs["model"])]
    groups = (
        ("ethnicity", config.ETHNICITIES, style.ETHNICITY_FAMILY),
        ("gender", config.GENDERS, style.GENDER_FAMILY),
        ("age", config.AGES, style.AGE_FAMILY),
    )
    slots, colours, handles = [], {}, []
    for g, (attribute, preferred, family) in enumerate(groups):
        levels = _levels(coefs, attribute, preferred)
        for level, colour in zip(levels, style.tint_series(family, len(levels))):
            slots.append((attribute, level))
            colours[(attribute, level)] = colour
            handles.append(Patch(color=colour, label=level))
        if g < len(groups) - 1:
            slots.append(None)
            handles.append(Line2D([], [], color="none", label=" "))

    fig, axes, ncols = _panel_grid(len(scenarios), len(models), ncols)
    for idx, scenario in enumerate(scenarios):
        _draw_effect_bars(axes.flat[idx], coefs[coefs["scenario"] == scenario], models, slots, colours, alpha)
    _finish_panels(fig, axes, scenarios, models, ncols)
    axes.flat[0].legend(
        handles=handles, title="E / G / A", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
        fontsize=14, title_fontsize=15,
    )
    return fig


def plot_attribute_effects(coefs: pd.DataFrame, models, attributes, scenarios=config.SCENARIOS, ncols: int = 2,
                           alpha=config.ME_ALPHA):
    """Marginal effects of appearance attributes per model, one panel per scenario."""
    style.apply_style()
    scenarios = [s for s in scenarios if s in set(coefs["scenario"])]
    models = [m for m in models if m in set(coefs["model"])]
    slots, colours, handles, labels = [], {}, [], []
    cursor = 0
    for a, attribute in enumerate(attributes):
        levels = _levels(coefs, attribute)
        reference = coefs.loc[coefs["attribute"] == attribute, "reference"].iloc[0]
        if a > 0:
            slots.append(None)
            handles.append(Line2D([], [], color="none"))
            labels.append(" ")
        handles.append(Line2D([], [], color="none"))
        labels.append(f"— {attribute_label(attribute)} —\n(ref: {level_label(attribute, reference)})")
        for level in levels:
            colour = ATTRIBUTE_COLOURS[cursor % len(ATTRIBUTE_COLOURS)]
            cursor += 1
            slots.append((attribute, level))
            colours[(attribute, level)] = colour
            handles.append(Patch(color=colour))
            labels.append(level_label(attribute, level))

    fig, axes, ncols = _panel_grid(len(scenarios), len(models), ncols)
    for idx, scenario in enumerate(scenarios):
        _draw_effect_bars(axes.flat[idx], coefs[coefs["scenario"] == scenario], models, slots, colours, alpha)
    _finish_panels(fig, axes, scenarios, models, ncols)
    for ax in axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=25)
    legend = axes.flat[0].legend(
        handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=13,
        handlelength=1.2,
    )
    for text, label in zip(legend.get_texts(), labels):
        if label.startswith("—"):
            text.set_fontweight("bold")
    return fig


def _standardized_residuals(fit) -> np.ndarray:
    results = fit.results
    if fit.estimator == "ols":
        return np.asarray(results.get_influence().resid_studentized_internal)
    return np.asarray(results.resid) / results.scale


def _annotate_top(ax, x, y, values, n=3):
    for i in np.argsort(-np.abs(values))[:n]:
        ax.annotate(str(i), xy=(x[i], y[i]), color="C3", fontsize=8)


def plot_residual_diagnostics(fits, titles):
    """Residuals vs fitted, normal Q-Q and scale-location plots, one column per fit."""
    style.apply_style()
    fig, axes = plt.subplots(3, len(fits), figsize=(4.2 * len(fits), 11), squeeze=False)
    for col, (fit, title) in enumerate(zip(fits, titles)):
        fitted = np.asarray(fit.results.fittedvalues)
        resid = np.asarray(fit.results.resid)
        std_resid = _standardized_residuals(fit)
        delta = 0.01 * np.ptp(fitted)

        ax = axes[0, col]
        ax.scatter(fitted, resid, alpha=0.5, s=10)
        smooth = lowess(resid, fitted, delta=delta)
        ax.plot(smooth[:, 0], smooth[:, 1], color="red", linewidth=1)
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
        _annotate_top(ax, fitted, resid, resid)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")

        ax = axes[1, col]
        order = np.argsort(std_resid)
        theoretical = stats.norm.ppf((np.arange(1, len(std_resid) + 1) - 0.5) / len(std_resid))
        ax.scatter(theoretical, std_resid[order], alpha=0.5, s=10)
        lims = [min(theoretical.min(), std_resid.min()), max(theoretical.max(), std_resid.max())]
        ax.plot(lims, lims, color="red", linewidth=1)
        positions = np.empty(len(order), dtype=int)
        positions[order] = np.arange(len(order))
        for i in np.argsort(-np.abs(std_resid))[:3]:
            ax.annotate(str(i), xy=(theoretical[positions[i]], std_resid[i]), color="C3", fontsize=8, ha="right")
        ax.set_title("Normal Q-Q", fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Standardized Residuals")

        ax = axes[2, col]
        root = np.sqrt(np.abs(std_resid))
        ax.scatter(fitted, root, alpha=0.5, s=10)
        smooth = lowess(root, fitted, delta=delta)
        ax.plot(smooth[:, 0], smooth[:, 1], color="red", linewidth=1)
        _annotate_top(ax, fitted, root, root)
        ax.set_title("Scale-Location", fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$")
    fig.tight_layout()
    return fig
