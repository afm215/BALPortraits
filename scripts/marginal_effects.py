"""Marginal effects of demographic and face attributes on portrait scores.

For every model and scenario, the portrait score is regressed on dummy-encoded
attributes with a robust linear regression (Huber M-estimator, IRLS):
  - controlled setup: ethnicity, gender, age and the 13 appearance categories;
  - uncontrolled setup: ethnicity, gender and age only.
References: Caucasian, Male and 40 for the demographic attributes, the most
frequent level (False for binary attributes) for the appearance categories.
Bars are opaque when p ≤ 0.05 and transparent otherwise; error bars are the
half-width of the 95% confidence interval.

Figures (file names of the paper):
  marginal_effects                       activities + occupations, 7 selected VLMs, controlled
  eth_gen_age_marginal_effects           4 scenarios, 7 selected VLMs, controlled
  eth_gen_age_marginal_effects_all_models  4 scenarios, all models, controlled
  eth_gen_age_marginal_effects_no_control  4 scenarios, 7 selected VLMs, uncontrolled
  <group>_marginal_effects               appearance attributes, all models
  residual_plot_not_fixed / _fixed       OLS residual diagnostics, occupations (--diagnostics)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from balportrait import cli, config, regression  # noqa: E402
from balportrait.plotting import marginal_effects as me_plots  # noqa: E402
from balportrait.plotting import style  # noqa: E402

CONTROLLED = config.CONTROL_ATTRIBUTES + config.DEMOGRAPHIC_ATTRIBUTES
UNCONTROLLED = config.DEMOGRAPHIC_ATTRIBUTES


def main():
    parser = cli.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--estimator", choices=["rlm", "ols"], default="rlm", help="regression estimator")
    parser.add_argument(
        "--drop-bottom-quantile", type=float, default=config.ME_DROP_BOTTOM_QUANTILE,
        help="exclude portraits whose score is not above this quantile (0 keeps all)",
    )
    parser.add_argument("--p-adjust", default=config.ME_P_ADJUST,
                        help="p-value adjustment within each fit ('none' or 'fdr_bh')")
    parser.add_argument("--alpha", type=float, default=config.ME_ALPHA, help="significance threshold of the figures")
    parser.add_argument("--diagnostics", action="store_true",
                        help="also draw OLS residual diagnostics (occupations) and write fit statistics")
    args = parser.parse_args()

    models = cli.resolve_models(args)
    scores = cli.load_scores(args, models)
    tables, figures = cli.tables_dir(args), cli.figures_dir(args)
    fit_kw = dict(estimator=args.estimator, drop_bottom_quantile=args.drop_bottom_quantile,
                  p_adjust=args.p_adjust, alpha=args.alpha)

    controlled, controlled_fits = regression.fit_all(scores, CONTROLLED, keep_fits=args.diagnostics, **fit_kw)
    uncontrolled, _ = regression.fit_all(scores, UNCONTROLLED, **fit_kw)
    cli.write_table(controlled, tables / "marginal_effects_controlled.csv", index=False)
    cli.write_table(uncontrolled, tables / "marginal_effects_uncontrolled.csv", index=False)
    impact = regression.control_impact_table(controlled, uncontrolled, alpha=args.alpha)
    cli.write_table(impact, tables / "control_impact.csv", index=False)

    ordered = config.order_models(models)
    selected = cli.figure_models(ordered, config.SELECTED_MODELS)
    paths = []
    demographic = [
        ("marginal_effects", controlled, selected, ("activities", "occupations")),
        ("eth_gen_age_marginal_effects", controlled, selected, ("banking", "activities", "brands", "occupations")),
        ("eth_gen_age_marginal_effects_all_models", controlled, ordered,
         ("banking", "activities", "brands", "occupations")),
        ("eth_gen_age_marginal_effects_no_control", uncontrolled, selected,
         ("banking", "activities", "brands", "occupations")),
    ]
    for stem, coefs, figure_models, scenarios in demographic:
        if not figure_models or not set(scenarios) & set(coefs["scenario"]):
            continue
        fig = me_plots.plot_demographic_effects(coefs, figure_models, scenarios=scenarios, alpha=args.alpha)
        paths += style.save_figure(fig, figures / stem, args.formats)
    for stem, attributes in config.ATTRIBUTE_FIGURE_GROUPS.items():
        fig = me_plots.plot_attribute_effects(
            controlled, ordered, attributes, scenarios=("banking", "activities", "brands", "occupations"),
            alpha=args.alpha,
        )
        paths += style.save_figure(fig, figures / f"{stem}_marginal_effects", args.formats)

    if args.diagnostics:
        stats = [
            {"model": model, "scenario": scenario, **regression.regression_diagnostics(fit)}
            for (model, scenario), fit in controlled_fits.items()
        ]
        cli.write_table(pd.DataFrame(stats), tables / "regression_diagnostics.csv", index=False)
        diag_models = [m for m in selected if "score_occupations" in scores[m]]
        for stem, quantile in (("residual_plot_not_fixed", None), ("residual_plot_fixed", args.drop_bottom_quantile)):
            fits = [
                regression.fit_marginal_effects(
                    scores[m], "score_occupations", CONTROLLED, estimator="ols", drop_bottom_quantile=quantile
                )
                for m in diag_models
            ]
            fig = me_plots.plot_residual_diagnostics(fits, [config.display_name(m) for m in diag_models])
            paths += style.save_figure(fig, figures / stem, args.formats)
    cli.report_figures(paths)


if __name__ == "__main__":
    main()
