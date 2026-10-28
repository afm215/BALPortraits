"""Standard deviation of the portrait scores per model and scenario.

Reproduces the fitness-score analysis figure of the paper
(``overall_prestige_std_by_model_experience``), which checks that the degree of
bias is not a by-product of a wider range of output scores.
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
    dispersion = bias_metrics.score_dispersion_table(scores)
    cli.write_table(dispersion, cli.tables_dir(args) / "score_dispersion.csv", float_format="%.5f")
    fig = bias_plots.plot_score_dispersion(dispersion, config.order_models(models))
    cli.report_figures(
        style.save_figure(fig, cli.figures_dir(args) / "overall_prestige_std_by_model_experience", args.formats)
    )


if __name__ == "__main__":
    main()
