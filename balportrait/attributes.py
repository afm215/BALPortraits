"""Dataset-level statistics of the categorical attributes: counts, Cramér's V, phi coefficients.

All functions work on :func:`balportrait.io.load_metadata`; ``"None"`` (absent or
not inferable value) is treated as a regular level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from . import config

# Order of the attributes in the association figures: appearance first, then demographics.
FIGURE_ORDER = config.CONTROL_ATTRIBUTES + ("ethnicity", "gender", "age")

CATEGORY_LABELS = {"moustache_style": "Mustache style", "hairstyle_texture_or_arrangement": "Hairstyle texture or arrangement"}


def category_label(attribute: str) -> str:
    return CATEGORY_LABELS.get(attribute, attribute.replace("_", " ").capitalize())


def attribute_counts(metadata: pd.DataFrame, attributes=config.ALL_ATTRIBUTES) -> pd.DataFrame:
    """Number of portraits per attribute value, ``None`` excluded, most frequent first."""
    records = []
    for attribute in attributes:
        counts = metadata[attribute].astype(str).value_counts()
        for value, count in counts.items():
            if value != "None":
                records.append({"category": attribute, "value": value, "count": int(count)})
    return pd.DataFrame(records)


def counts_to_latex(counts: pd.DataFrame) -> str:
    """Body of the attribute table (``supertabular`` rows, categories in alphabetical order)."""
    lines = []
    for attribute in sorted(counts["category"].unique(), key=category_label):
        rows = counts[counts["category"] == attribute]
        lines.append(f"% --- {category_label(attribute).upper()} ---")
        for i, (_, row) in enumerate(rows.iterrows()):
            value = {"True": "Present", "False": "Absent"}.get(row["value"], row["value"]).replace("_", r"\_")
            label = category_label(attribute) if i == 0 else ""
            lines.append(f"{label} & {value} & {row['count']} \\\\")
        lines.append(r"\midrule")
        lines.append("")
    return "\n".join(lines[:-2]) + "\n"


def cramers_v_corrected(x, y) -> float:
    """Bias-corrected Cramér's V (Bergsma, 2013).

    ``Ṽ = sqrt(φ̃² / min(k̃ - 1, r̃ - 1))`` with ``φ̃² = max(0, φ² - (k-1)(r-1)/(n-1))``,
    ``k̃ = k - (k-1)²/(n-1)`` and ``r̃ = r - (r-1)²/(n-1)``.
    """
    table = pd.crosstab(pd.Series(x).to_numpy(), pd.Series(y).to_numpy())
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_tilde = r - (r - 1) ** 2 / (n - 1)
    k_tilde = k - (k - 1) ** 2 / (n - 1)
    denominator = min(k_tilde - 1, r_tilde - 1)
    return float(np.sqrt(phi2 / denominator)) if denominator > 0 else 0.0


def cramers_v_matrix(metadata: pd.DataFrame, attributes=FIGURE_ORDER) -> pd.DataFrame:
    """Symmetric matrix of corrected Cramér's V between attributes (1 on the diagonal)."""
    attributes = list(attributes)
    matrix = pd.DataFrame(np.eye(len(attributes)), index=attributes, columns=attributes)
    for i, a in enumerate(attributes):
        for b in attributes[i + 1:]:
            matrix.loc[a, b] = matrix.loc[b, a] = cramers_v_corrected(metadata[a].astype(str), metadata[b].astype(str))
    return matrix


def strong_pairs(cramers: pd.DataFrame, threshold: float = 0.3, exclude=("eye_color",)):
    """Attribute pairs with Ṽ above ``threshold`` (strongest first), skipping ``exclude``."""
    names = list(cramers.index)
    pairs = [
        (a, b, float(cramers.loc[a, b]))
        for i, a in enumerate(names)
        for b in names[i + 1:]
        if cramers.loc[a, b] > threshold and a not in exclude and b not in exclude
    ]
    return sorted(pairs, key=lambda p: -abs(p[2]))


def one_hot(series: pd.Series) -> pd.DataFrame:
    """Indicator column per level (sorted levels)."""
    values = series.astype(str)
    return pd.DataFrame({level: (values == level).astype(float) for level in sorted(values.unique())})


def phi_matrix(metadata: pd.DataFrame, attr_a: str, attr_b: str) -> pd.DataFrame:
    """Phi coefficient (Pearson correlation of indicators) for every pair of levels."""
    a, b = one_hot(metadata[attr_a]), one_hot(metadata[attr_b])
    matrix = pd.DataFrame(index=a.columns, columns=b.columns, dtype=float)
    for la in a.columns:
        for lb in b.columns:
            sa, sb = a[la].std(), b[lb].std()
            matrix.loc[la, lb] = np.nan if sa == 0 or sb == 0 else np.cov(a[la], b[lb])[0, 1] / (sa * sb)
    return matrix


def phi_pairs_table(metadata: pd.DataFrame, pairs, min_abs_phi: float = 0.0) -> pd.DataFrame:
    """Long table of phi coefficients of the given pairs, with joint counts and lift."""
    records = []
    for attr_a, attr_b, v in pairs:
        a, b = one_hot(metadata[attr_a]), one_hot(metadata[attr_b])
        phi = phi_matrix(metadata, attr_a, attr_b)
        for la in a.columns:
            for lb in b.columns:
                value = phi.loc[la, lb]
                if np.isnan(value) or abs(value) < min_abs_phi:
                    continue
                joint = a[la] * b[lb]
                expected = a[la].mean() * b[lb].mean()
                records.append(
                    {
                        "var_1": attr_a, "val_1": la, "var_2": attr_b, "val_2": lb, "cramers_v": v, "phi": value,
                        "joint_count": int(joint.sum()), "lift": joint.mean() / expected if expected else np.nan,
                    }
                )
    table = pd.DataFrame(records)
    return table.reindex(table["phi"].abs().sort_values(ascending=False).index).reset_index(drop=True)
