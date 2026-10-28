"""Portrait prestige scores and ranks of every model, exported as one CSV per model.

For each scenario S, fitness scores are min-max normalised per item, weighted by the
standardised item prestige and averaged over items:

    a_S = 1/|S| · Σ_i (p_i − μ_p)/σ_p · f̂_i

Portraits are then ranked by a_S (rank 1 = highest score). The global rank is the
mean of the four scenario ranks.

Output: scores/<model>.csv with HASH, ethnicity, gender, age, score_<scenario>,
rank_<scenario> and global_rank.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balportrait import cli, config  # noqa: E402


def main():
    parser = cli.base_parser(__doc__.splitlines()[0])
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model, scores in cli.load_scores(args).items():
        columns = ["ethnicity", "gender", "age"]
        columns += [c for s in config.SCENARIOS for c in (f"score_{s}", f"rank_{s}") if c in scores]
        columns += ["global_rank"] if "global_rank" in scores else []
        cli.write_table(scores[columns], out_dir / f"{model}.csv")


if __name__ == "__main__":
    main()
