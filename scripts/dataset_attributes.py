"""Metrics characterising the dataset attributes: counts, Cramér's V, phi coefficients, VIF.

These statistics depend only on the dataset metadata (demographics and the 13
appearance categories), except the VIF, which is computed on the design matrix
of the controlled marginal-effect regression.

Outputs:
  tables/attribute_counts.{csv,tex}  values and counts of every attribute ("None" excluded)
  tables/cramers_v.csv               bias-corrected Cramér's V between attribute pairs
  tables/phi_pairs.csv               phi coefficients of the pairs with Cramér's V > threshold
  tables/vif.csv                     VIF of every dummy-encoded predictor
  figures/cramers_v_matrix_corrected, figures/phi_correlations_aggregated, figures/vif_scores
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balportrait import attributes, cli, config, io, regression, scoring  # noqa: E402
from balportrait.plotting import attributes as attribute_plots  # noqa: E402
from balportrait.plotting import style  # noqa: E402


def main():
    parser = cli.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--phi-threshold", type=float, default=0.3, help="Cramér's V threshold of the phi figure")
    parser.add_argument("--phi-exclude", nargs="*", default=["eye_color"], help="attributes left out of the phi figure")
    parser.add_argument(
        "--vif-model", default="gpt54mini",
        help="model whose fit rows define the VIF design matrix ('none': all 9,000 portraits)",
    )
    parser.add_argument("--vif-scenario", default="occupations", choices=config.SCENARIOS)
    parser.add_argument("--drop-bottom-quantile", type=float, default=config.ME_DROP_BOTTOM_QUANTILE,
                        help="bottom-quantile filter of the VIF design (as in the marginal-effect fits)")
    args = parser.parse_args()

    metadata = io.load_metadata()
    tables, figures = cli.tables_dir(args), cli.figures_dir(args)

    counts = attributes.attribute_counts(metadata)
    cli.write_table(counts, tables / "attribute_counts.csv", index=False)
    (tables / "attribute_counts.tex").write_text(attributes.counts_to_latex(counts))
    print(f"wrote {tables / 'attribute_counts.tex'}")
    controls = counts[~counts["category"].isin(config.DEMOGRAPHIC_ATTRIBUTES)]
    print(f"{controls['category'].nunique()} appearance categories, {len(controls)} values")

    cramers = attributes.cramers_v_matrix(metadata)
    cli.write_table(cramers, tables / "cramers_v.csv", float_format="%.4f")
    pairs = attributes.strong_pairs(cramers, threshold=args.phi_threshold, exclude=tuple(args.phi_exclude))
    cli.write_table(attributes.phi_pairs_table(metadata, pairs), tables / "phi_pairs.csv", index=False,
                    float_format="%.4f")

    if args.vif_model.lower() == "none":
        design, _ = regression.build_design_matrix(metadata, regression_attributes())
    else:
        scores = scoring.compute_model_scores(args.outputs_dir, args.vif_model, metadata, scenarios=[args.vif_scenario],
                                              verbose=not args.quiet)
        fit = regression.fit_marginal_effects(scores, f"score_{args.vif_scenario}", regression_attributes(),
                                              drop_bottom_quantile=args.drop_bottom_quantile)
        design = fit.X
    vif = regression.vif_scores(design)
    cli.write_table(vif.to_frame(), tables / "vif.csv", float_format="%.4f")
    print(f"max VIF: {vif.max():.3f} ({vif.idxmax()})")

    paths = style.save_figure(attribute_plots.plot_cramers_v(cramers), figures / "cramers_v_matrix_corrected",
                              args.formats)
    panels = [(a, b, v, attributes.phi_matrix(metadata, a, b)) for a, b, v in pairs]
    if panels:
        paths += style.save_figure(attribute_plots.plot_phi_panels(panels), figures / "phi_correlations_aggregated",
                                   args.formats)
    paths += style.save_figure(attribute_plots.plot_vif(vif), figures / "vif_scores", args.formats)
    cli.report_figures(paths)


def regression_attributes():
    return config.CONTROL_ATTRIBUTES + config.DEMOGRAPHIC_ATTRIBUTES


if __name__ == "__main__":
    main()
