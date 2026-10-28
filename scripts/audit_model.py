"""Audit one or more new VLMs on BALPortrait with the full paper protocol.

Expects the outputs of the new model(s) at <outputs-dir>/<scenario>/<model>.csv[.gz]
(see scripts/prepare_model_outputs.py). Writes into <results-dir>/audit_<model>/:
  tables/  degree of bias (+ equity line), group mean ranks, pairwise Mann-Whitney U
           tests with rank-biserial correlations, controlled/uncontrolled marginal effects
  figures/ degree of bias (next to the 16 paper models with --compare-with paper),
           one-way ranks, intersectional matrix, significance heatmaps, marginal effects

Example:
    python scripts/audit_model.py --model my_vlm --outputs-dir my_outputs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from balportrait import bias_metrics, cli, config, io, regression, scoring, significance  # noqa: E402
from balportrait.plotting import bias as bias_plots  # noqa: E402
from balportrait.plotting import marginal_effects as me_plots  # noqa: E402
from balportrait.plotting import significance as significance_plots  # noqa: E402
from balportrait.plotting import style  # noqa: E402


def main():
    parser = cli.base_parser(__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", required=True, help="model(s) to audit")
    parser.add_argument("--compare-with", choices=["paper", "none"], default="paper",
                        help="show the 16 audited models of the paper next to the new model(s)")
    parser.add_argument("--paper-outputs-dir", type=Path, default=config.DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--null-seeds", type=int, default=20, help="random-score runs of the equity line")
    args = parser.parse_args()

    metadata = io.load_metadata()
    new = {m: scoring.compute_model_scores(args.outputs_dir, m, metadata, verbose=not args.quiet) for m in args.model}
    reference = {}
    if args.compare_with == "paper":
        paper_models = [m for m in io.discover_models(args.paper_outputs_dir) if m in config.PAPER_MODELS]
        reference = scoring.compute_all_scores(args.paper_outputs_dir, paper_models, metadata, verbose=False)

    root = Path(args.results_dir) / f"audit_{'_'.join(args.model)}"
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    paths = []

    # Degree of bias, equity line, significance pies
    everything = {**reference, **new}
    dob = bias_metrics.dob_table(everything)
    cli.write_table(dob, tables / "degree_of_bias.csv", float_format="%.2f")
    equity_line = None
    if args.null_seeds > 0:
        null = bias_metrics.null_baseline(metadata, n_seeds=args.null_seeds)
        equity_line = float(null["dob"].mean())
        print(f"equity line: {equity_line:.1f} ± {null['dob'].std():.1f} ranks")
    summary = significance.global_significance_summary(everything)
    if not summary.empty:
        cli.write_table(summary, tables / "global_mwu_summary.csv", float_format="%.4f")
    fig = bias_plots.plot_degree_of_bias(dob, summary if not summary.empty else None, equity_line, list(everything))
    paths += style.save_figure(fig, figures / "degree_of_bias", args.formats)

    # Group mean ranks
    one_way = bias_metrics.mean_rank_table(new, by=("ethnicity", "age", "gender"))
    ega = bias_metrics.mean_rank_table(new, by=("ega",))
    cli.write_table(one_way, tables / "mean_rank_one_way.csv", index=False, float_format="%.2f")
    cli.write_table(ega, tables / "mean_rank_ega.csv", index=False, float_format="%.2f")
    paths += style.save_figure(bias_plots.plot_one_way_ranks(one_way, list(new)), figures / "one_way_ranks",
                               args.formats)
    paths += style.save_figure(bias_plots.plot_intersectional_matrices(ega, list(new)),
                               figures / "intersectional_ranks", args.formats)

    # Pairwise tests
    pairwise = significance.pairwise_tables(new)
    cli.write_table(pairwise, tables / "pairwise_mwu.csv", index=False)
    cli.write_table(significance.mean_abs_rrb_table(pairwise), tables / "mean_abs_rrb_ega.csv", float_format="%.4f")
    for value in ("p_value", "r_rb"):
        paths += style.save_figure(significance_plots.plot_pairwise_heatmaps(pairwise, list(new), value=value),
                                   figures / f"pairwise_one_way_{value}", args.formats)
        paths += style.save_figure(significance_plots.plot_joint_heatmaps(pairwise, list(new), value=value),
                                   figures / f"pairwise_ega_{value}", args.formats)

    # Marginal effects
    controlled, _ = regression.fit_all(new, config.CONTROL_ATTRIBUTES + config.DEMOGRAPHIC_ATTRIBUTES)
    uncontrolled, _ = regression.fit_all(new, config.DEMOGRAPHIC_ATTRIBUTES)
    cli.write_table(controlled, tables / "marginal_effects_controlled.csv", index=False)
    cli.write_table(uncontrolled, tables / "marginal_effects_uncontrolled.csv", index=False)
    cli.write_table(regression.control_impact_table(controlled, uncontrolled), tables / "control_impact.csv",
                    index=False)
    # Demographic effects of the new model(s) next to the 7 selected paper VLMs
    comparison = [m for m in config.SELECTED_MODELS if m in reference] + list(new)
    demographic = controlled
    if reference:
        controlled_reference, _ = regression.fit_all(
            {m: reference[m] for m in config.SELECTED_MODELS if m in reference},
            config.CONTROL_ATTRIBUTES + config.DEMOGRAPHIC_ATTRIBUTES,
        )
        demographic = pd.concat([controlled_reference, controlled], ignore_index=True)
    paths += style.save_figure(me_plots.plot_demographic_effects(demographic, comparison),
                               figures / "eth_gen_age_marginal_effects", args.formats)
    paths += style.save_figure(me_plots.plot_demographic_effects(uncontrolled, list(new)),
                               figures / "eth_gen_age_marginal_effects_no_control", args.formats)
    for stem, attributes in config.ATTRIBUTE_FIGURE_GROUPS.items():
        paths += style.save_figure(me_plots.plot_attribute_effects(controlled, list(new), attributes),
                                   figures / f"{stem}_marginal_effects", args.formats)
    cli.report_figures(paths)


if __name__ == "__main__":
    main()
