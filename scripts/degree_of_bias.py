"""Degree of bias (DoB) per model and scenario, equity line and Mann-Whitney U pies.

Reproduces the degree-of-bias figure of the paper (``global_experiment_ranks_equity_line``):
the DoB is the standard deviation of the mean ranks of the 24 ethnicity-gender-age
(EGA) segments; the global DoB is its mean over the four scenarios. The equity line
is the DoB obtained when every fitness score is replaced by a random score. The pies
summarise the 276 pairwise Mann-Whitney U tests between EGA segments on the global
rank of each portrait, with Benjamini-Hochberg adjustment within each model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balportrait import bias_metrics, cli, config, io, significance  # noqa: E402
from balportrait.plotting import bias as bias_plots  # noqa: E402
from balportrait.plotting import style  # noqa: E402


def main():
    parser = cli.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--null-seeds", type=int, default=20, help="random-score runs of the null baseline")
    parser.add_argument("--seed", type=int, default=0, help="seed of the null baseline")
    parser.add_argument("--p-adjust", default="fdr_bh", help="p-value adjustment of the pies ('none' or 'fdr_bh')")
    args = parser.parse_args()

    models = cli.resolve_models(args)
    scores = cli.load_scores(args, models)
    tables, figures = cli.tables_dir(args), cli.figures_dir(args)

    dob = bias_metrics.dob_table(scores)
    cli.write_table(dob, tables / "degree_of_bias.csv", float_format="%.2f")

    equity_line = None
    if args.null_seeds > 0:
        null = bias_metrics.null_baseline(io.load_metadata(), n_seeds=args.null_seeds, seed=args.seed)
        cli.write_table(null, tables / "null_baseline.csv", index=False, float_format="%.2f")
        equity_line = float(null["dob"].mean())
        print(
            f"equity line: {equity_line:.1f} ± {null['dob'].std():.1f} ranks "
            f"(closed form {bias_metrics.expected_null_dob():.1f})"
        )

    summary = significance.global_significance_summary(scores, p_adjust=args.p_adjust)
    if not summary.empty:
        cli.write_table(summary, tables / "global_mwu_summary.csv", float_format="%.4f")

    fig = bias_plots.plot_degree_of_bias(
        dob, summary if not summary.empty else None, equity_line, config.order_models(models)
    )
    cli.report_figures(style.save_figure(fig, figures / "global_experiment_ranks_equity_line", args.formats))


if __name__ == "__main__":
    main()
