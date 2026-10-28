"""Marginal effects of demographic and appearance attributes on portrait scores.

The portrait score of a scenario is regressed on dummy-encoded categorical
attributes. With binary regressors, each coefficient is the expected score change
when switching from the reference level to the given level, all other attributes
being equal. Estimation uses robust linear regression (Huber M-estimator solved by
IRLS, statsmodels ``RLM``) by default, or OLS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.robust.scale as robust_scale
import statsmodels.stats.stattools as stattools
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config, scoring
from .significance import adjust_pvalues


def _levels(series: pd.Series) -> pd.Series:
    """Attribute values as title-cased strings (``black_hair`` → ``Black_Hair``, ``None`` kept)."""
    return series.astype(str).str.title()


def reference_level(series: pd.Series, attribute: str, references=None) -> str:
    """Reference level of an attribute.

    Fixed reference if given in ``references``, ``False`` for boolean attributes, and
    the most frequent level otherwise.
    """
    levels = _levels(series)
    if references and attribute in references:
        return str(references[attribute]).title()
    if set(levels.unique()) <= {"True", "False"}:
        return "False"
    return levels.value_counts().idxmax()


def build_design_matrix(df: pd.DataFrame, attributes, references=config.DEMOGRAPHIC_REFERENCES):
    """Dummy-encode ``attributes`` of ``df`` with a constant column.

    Returns
    -------
    X : pd.DataFrame
        ``const`` followed by one column ``<attribute>_<Level>`` per non-reference level.
    refs : dict
        Reference level of every attribute.
    """
    columns, refs = [], {}
    for attribute in attributes:
        levels = _levels(df[attribute])
        ref = reference_level(df[attribute], attribute, references)
        if ref not in set(levels):
            raise ValueError(f"Reference level '{ref}' of '{attribute}' does not occur in the data")
        dummies = pd.get_dummies(levels, prefix=attribute, dtype=float).drop(columns=f"{attribute}_{ref}")
        columns.append(dummies)
        refs[attribute] = ref
    X = sm.add_constant(pd.concat(columns, axis=1), has_constant="add")
    return X, refs


def split_term(term: str, attributes):
    """``"hair_color_Black_Hair"`` → ``("hair_color", "Black_Hair")``."""
    for attribute in sorted(attributes, key=len, reverse=True):
        if term.startswith(f"{attribute}_"):
            return attribute, term[len(attribute) + 1:]
    return None, term


@dataclass
class MarginalEffectFit:
    """A fitted marginal-effect regression and its design."""

    results: object
    X: pd.DataFrame
    y: pd.Series
    references: dict
    attributes: tuple
    estimator: str

    @property
    def n_obs(self) -> int:
        return len(self.y)

    def coefficients(self, alpha: float = 0.05, p_adjust: str = "none") -> pd.DataFrame:
        """Coefficient table (constant excluded).

        ``half_width`` is half the width of the (1 - alpha) confidence interval, used as
        error bar in the figures. ``p_adj`` is adjusted over all coefficients of the fit.
        """
        ci = self.results.conf_int(alpha=alpha)
        params = self.results.params
        terms = [t for t in params.index if t != "const"]
        table = pd.DataFrame(
            {
                "term": terms,
                "estimate": params[terms].to_numpy(),
                "std_err": self.results.bse[terms].to_numpy(),
                "ci_low": ci.loc[terms, 0].to_numpy(),
                "ci_high": ci.loc[terms, 1].to_numpy(),
                "p_value": self.results.pvalues[terms].to_numpy(),
            }
        )
        table["half_width"] = (table["ci_high"] - table["ci_low"]) / 2
        table["p_adj"] = adjust_pvalues(table["p_value"], p_adjust)
        parsed = [split_term(t, self.attributes) for t in terms]
        table.insert(1, "attribute", [a for a, _ in parsed])
        table.insert(2, "level", [lvl for _, lvl in parsed])
        table.insert(3, "reference", [self.references.get(a) for a, _ in parsed])
        table["n_obs"] = self.n_obs
        return table


def fit_marginal_effects(
    df: pd.DataFrame,
    target_col: str,
    attributes,
    estimator: str = "rlm",
    drop_bottom_quantile: float | None = config.ME_DROP_BOTTOM_QUANTILE,
    references=config.DEMOGRAPHIC_REFERENCES,
) -> MarginalEffectFit:
    """Regress ``target_col`` on the dummy-encoded ``attributes``.

    Parameters
    ----------
    estimator
        ``"rlm"``: Huber M-estimator (t = 1.345, MAD scale, H1 covariance, Wald z-tests).
        ``"ols"``: ordinary least squares with HC3 standard errors.
    drop_bottom_quantile
        If set, portraits whose score is not above this quantile of the scores are
        excluded before fitting (0 or None keeps every portrait).
    """
    data = df.dropna(subset=[target_col])
    if drop_bottom_quantile:
        data = data[data[target_col] > data[target_col].quantile(drop_bottom_quantile)]
    X, refs = build_design_matrix(data, attributes, references)
    y = data[target_col]
    if estimator == "rlm":
        results = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    elif estimator == "ols":
        results = sm.OLS(y, X).fit(cov_type="HC3")
    else:
        raise ValueError("estimator must be 'rlm' or 'ols'")
    return MarginalEffectFit(results, X, y, refs, tuple(attributes), estimator)


def fit_all(
    scores_by_model: dict,
    attributes,
    estimator: str = "rlm",
    drop_bottom_quantile: float | None = config.ME_DROP_BOTTOM_QUANTILE,
    p_adjust: str = config.ME_P_ADJUST,
    alpha: float = config.ME_ALPHA,
    keep_fits: bool = False,
):
    """Fit every model × scenario and gather the coefficients in one long table.

    Returns
    -------
    coefficients : pd.DataFrame
        One row per model, scenario and term.
    fits : dict
        ``{(model, scenario): MarginalEffectFit}`` (empty unless ``keep_fits``).
    """
    tables, fits = [], {}
    for model, scores in scores_by_model.items():
        for scenario in scoring.available_scenarios(scores):
            fit = fit_marginal_effects(
                scores, f"score_{scenario}", attributes, estimator=estimator,
                drop_bottom_quantile=drop_bottom_quantile,
            )
            table = fit.coefficients(alpha=alpha, p_adjust=p_adjust)
            table.insert(0, "scenario", scenario)
            table.insert(0, "model", model)
            tables.append(table)
            if keep_fits:
                fits[(model, scenario)] = fit
    return pd.concat(tables, ignore_index=True), fits


def control_impact_table(controlled: pd.DataFrame, uncontrolled: pd.DataFrame, alpha: float = config.ME_ALPHA):
    """Demographic effects with and without appearance controls.

    Flags ``sign_reversal`` (both significant, opposite signs) and
    ``significance_lost`` (significant without control, not with control), using
    the ``p_adj`` column of both tables.
    """
    keys = ["model", "scenario", "term", "attribute", "level", "reference"]
    columns = keys + ["estimate", "p_adj"]
    demographic = controlled["attribute"].isin(config.DEMOGRAPHIC_ATTRIBUTES)
    merged = controlled.loc[demographic, columns].merge(
        uncontrolled[columns], on=keys, suffixes=("_controlled", "_uncontrolled")
    )
    merged["delta"] = merged["estimate_controlled"] - merged["estimate_uncontrolled"]
    sig_c = merged["p_adj_controlled"] < alpha
    sig_u = merged["p_adj_uncontrolled"] < alpha
    merged["sign_reversal"] = sig_c & sig_u & (np.sign(merged["estimate_controlled"]) != np.sign(merged["estimate_uncontrolled"]))
    merged["significance_lost"] = sig_u & ~sig_c
    return merged


def vif_scores(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor of every non-constant column of a design matrix."""
    values = X.to_numpy(dtype=float)
    vif = {col: variance_inflation_factor(values, i) for i, col in enumerate(X.columns) if col != "const"}
    return pd.Series(vif, name="vif").sort_values(ascending=False)


def regression_diagnostics(fit: MarginalEffectFit) -> dict:
    """Goodness-of-fit and residual statistics of a fit."""
    resid = np.asarray(fit.results.resid)
    y = np.asarray(fit.y)
    jb, _, skew, kurtosis = stattools.jarque_bera(resid)
    omnibus, _ = stattools.omni_normtest(resid)
    singular = np.linalg.svd(fit.X.to_numpy(dtype=float), compute_uv=False)
    record = {
        "estimator": fit.estimator,
        "n_obs": fit.n_obs,
        "n_terms": fit.X.shape[1] - 1,
        "jarque_bera": float(jb),
        "omnibus": float(omnibus),
        "skew": float(skew),
        "kurtosis": float(kurtosis),
        "durbin_watson": float(stattools.durbin_watson(resid)),
        "condition_number": float(singular.max() / singular.min()),
    }
    if fit.estimator == "ols":
        record["r_squared"] = float(fit.results.rsquared)
        record["adj_r_squared"] = float(fit.results.rsquared_adj)
    else:
        mad_y = robust_scale.mad(y - np.median(y))
        record["robust_r_squared_mad"] = float(1 - (robust_scale.mad(resid) / mad_y) ** 2) if mad_y else np.nan
    return record
