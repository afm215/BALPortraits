"""Mean portrait ranks per demographic group: one-way and intersectional figures.

Reproduces the one-way figures (``detailed_experiment_ranks`` for the 7 selected
VLMs, ``app_detailed_experiment_ranks`` for the others) and the intersectional
ethnicity-gender-age matrices (``detailed_frontier_matrices_ranks`` for the frontier
VLMs, ``app_detailed_frontier_matrices_ranks`` for the others). Lower ranks mean
higher prestige-weighted fitness.

Models outside the paper set (e.g. your own model) are added to the main figures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balportrait import bias_metrics, cli, config  # noqa: E402
from balportrait.plotting import bias as bias_plots  # noqa: E402
from balportrait.plotting import style  # noqa: E402


def main():
    parser = cli.base_parser(__doc__.splitlines()[0])
    args = parser.parse_args()

    models = cli.resolve_models(args)
    scores = cli.load_scores(args, models)
    tables, figures = cli.tables_dir(args), cli.figures_dir(args)

    one_way = bias_metrics.mean_rank_table(scores, by=("ethnicity", "age", "gender"))
    ega = bias_metrics.mean_rank_table(scores, by=("ega",))
    cli.write_table(one_way, tables / "mean_rank_one_way.csv", index=False, float_format="%.2f")
    cli.write_table(ega, tables / "mean_rank_ega.csv", index=False, float_format="%.2f")

    selected = cli.figure_models(models, config.SELECTED_MODELS)
    others = [m for m in models if m in config.PAPER_MODELS and m not in config.SELECTED_MODELS]
    frontier = cli.figure_models(models, config.FRONTIER_MODELS)
    non_frontier = [m for m in models if m in config.PAPER_MODELS and m not in config.FRONTIER_MODELS]

    paths = []
    if selected:
        fig = bias_plots.plot_one_way_ranks(one_way, selected)
        paths += style.save_figure(fig, figures / "detailed_experiment_ranks", args.formats)
    if others:
        fig = bias_plots.plot_one_way_ranks(one_way, others)
        paths += style.save_figure(fig, figures / "app_detailed_experiment_ranks", args.formats)
    if frontier:
        fig = bias_plots.plot_intersectional_matrices(ega, frontier)
        paths += style.save_figure(fig, figures / "detailed_frontier_matrices_ranks", args.formats)
    if non_frontier:
        fig = bias_plots.plot_intersectional_matrices(ega, non_frontier)
        paths += style.save_figure(fig, figures / "app_detailed_frontier_matrices_ranks", args.formats)
    cli.report_figures(paths)


if __name__ == "__main__":
    main()
