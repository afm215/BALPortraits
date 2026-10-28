"""Convert raw per-image answers of a VLM into the model-output table used by the analyses.

The input folder holds one ``<HASH>.txt`` file per portrait, containing the raw
answer of the model to the scenario prompt (``data/prompts/<scenario>.txt``): one
line ``ITEM:SCORE`` per item, where ITEM is the numeric item id (or, for brands, the
brand name). Invalid lines are ignored and items without a valid answer are left
empty; the validation report lists missing portraits and scores.

Example:
    python scripts/prepare_model_outputs.py --raw-dir answers/occupations \\
        --scenario occupations --model my_vlm
writes data/model_outputs/occupations/my_vlm.csv.gz.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balportrait import config, io  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="folder of <HASH>.txt answers")
    parser.add_argument("--scenario", required=True, choices=config.SCENARIOS)
    parser.add_argument("--model", required=True, help="model identifier (output file name)")
    parser.add_argument("--outputs-dir", type=Path, default=config.DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--score-range", type=float, nargs=2, default=(1, 10), metavar=("MIN", "MAX"),
                        help="valid score range of the prompt")
    args = parser.parse_args()

    table = io.parse_raw_answers(args.raw_dir, args.scenario, score_range=tuple(args.score_range))
    report = io.validate_model_output(table, args.scenario, args.model, io.load_demographics().index)
    print(report.summary())
    n_incomplete = int(table.drop(columns="HASH").isna().any(axis=1).sum())
    if n_incomplete:
        print(f"  {n_incomplete} portraits have at least one missing item score")
    report.raise_if_fatal()

    path = Path(args.outputs_dir) / args.scenario / f"{args.model}.csv.gz"
    io.write_model_output(table, path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
