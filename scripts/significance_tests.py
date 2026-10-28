"""Pairwise Mann-Whitney U tests and rank-biserial correlations between demographic groups.

For every model and scenario, all pairs of gender, age, ethnicity and EGA
(ethnicity × gender × age) groups are compared on the portrait ranks with a
two-sided Mann-Whitney U test. The rank-biserial correlation r_rb = 1 - 2U/(n_A n_B)
is reported with its sign (r_rb > 0: group A is ranked better than group B).

Outputs:
  tables/pairwise_mwu.csv            every test (p-values and r_rb)
  tables/mean_abs_rrb_ega.csv        mean |r_rb| over the 276 EGA pairs per model and scenario
  tables/rrb_vs_dob_per_scenario.csv model orderings by mean |r_rb| and by degree of bias
  tables/rrb_vs_dob_agreement.csv    Spearman / Kendall agreement of the two orderings
  figures/gpt54min_claudehaiku_{p_value,r_rb}   one-way heatmaps (GPT54M and CLAH45)
  figures/mwu_joint_ega_{p_value,r_rb}          EGA heatmaps (GPT54M and CLAH45)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balportrait import bias_metrics, cli, config, significance  # noqa: E402
from balportrait.plotting import significance as significance_plots  # noqa: E402
from balportrait.plotting import style  # noqa: E402


def main():
    parser = cli.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--p-adjust", default="none", help="p-value adjustment ('none' or 'fdr_bh'), column p_adj")
    parser.add_argument(
        "--heatmap-models", nargs="+", default=None,
        help="models shown in the heatmaps (default: GPT54M, CLAH45 and any model outside the paper set)",
    )
    parser.add_argument("--latex", action="store_true", help="also write the summary tables as LaTeX")
    args = parser.parse_args()

    models = cli.resolve_models(args)
    scores = cli.load_scores(args, models)
    tables, figures = cli.tables_dir(args), cli.figures_dir(args)

    pairwise = significance.pairwise_tables(scores, p_adjust=args.p_adjust)
    cli.write_table(pairwise, tables / "pairwise_mwu.csv", index=False)

    mean_rrb = significance.mean_abs_rrb_table(pairwise, attribute="ega")
    cli.write_table(mean_rrb, tables / "mean_abs_rrb_ega.csv", float_format="%.4f")

    dob = bias_metrics.dob_table(scores)
    comparison, agreement = significance.ranking_agreement(mean_rrb, dob)
    cli.write_table(comparison, tables / "rrb_vs_dob_per_scenario.csv", index=False, float_format="%.4f")
    cli.write_table(agreement, tables / "rrb_vs_dob_agreement.csv", index=False)
    if len(agreement):
        print(f"mean Spearman rho over scenarios (r_rb vs DoB orderings): {agreement['spearman_rho'].mean():.3f}")

    if args.latex:
        for name, table in (("mean_abs_rrb_ega", mean_rrb), ("rrb_vs_dob_agreement", agreement.set_index("scenario"))):
            path = tables / f"{name}.tex"
            path.write_text(table.rename(index=config.display_name).to_latex(float_format="%.3f"))
            print(f"wrote {path}")

    heatmap_models = args.heatmap_models or cli.figure_models(models, config.SIGNIFICANCE_MODELS)
    heatmap_models = [m for m in heatmap_models if m in scores]
    if not heatmap_models:
        return
    paper_set = list(heatmap_models) == list(config.SIGNIFICANCE_MODELS)
    one_way_stem = "gpt54min_claudehaiku" if paper_set else "pairwise_one_way"
    joint_stem = "mwu_joint_ega" if paper_set else "pairwise_ega"
    paths = []
    for value in ("p_value", "r_rb"):
        fig = significance_plots.plot_pairwise_heatmaps(pairwise, heatmap_models, value=value)
        paths += style.save_figure(fig, figures / f"{one_way_stem}_{value}", args.formats)
        fig = significance_plots.plot_joint_heatmaps(pairwise, heatmap_models, value=value)
        paths += style.save_figure(fig, figures / f"{joint_stem}_{value}", args.formats)
    cli.report_figures(paths)


if __name__ == "__main__":
    main()
