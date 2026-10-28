"""Argument parsing and loading helpers shared by the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, io, scoring


def base_parser(description: str) -> argparse.ArgumentParser:
    """Parser with the options common to every analysis script."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--outputs-dir", type=Path, default=config.DEFAULT_OUTPUTS_DIR,
        help="folder with <scenario>/<model>.csv[.gz] model outputs",
    )
    parser.add_argument("--models", nargs="+", default=None, help="models to analyse (default: all found)")
    parser.add_argument("--results-dir", type=Path, default=config.DEFAULT_RESULTS_DIR, help="output folder")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="figure formats")
    parser.add_argument("--quiet", action="store_true", help="do not print data validation messages")
    return parser


def resolve_models(args) -> list[str]:
    """Models requested on the command line, or every model of ``--outputs-dir``."""
    available = io.discover_models(args.outputs_dir)
    if not available:
        raise SystemExit(f"No model output found in {args.outputs_dir}")
    if args.models is None:
        return list(available)
    missing = [m for m in args.models if m not in available]
    if missing:
        raise SystemExit(f"No output found for {missing} in {args.outputs_dir}")
    return config.order_models(args.models)


def load_scores(args, models=None) -> dict[str, pd.DataFrame]:
    """Portrait scores and ranks of every requested model."""
    metadata = io.load_metadata()
    return scoring.compute_all_scores(args.outputs_dir, models or resolve_models(args), metadata, verbose=not args.quiet)


def figure_models(requested: list[str], subset) -> list[str]:
    """The paper subset of models present in ``requested``; models outside the paper set are appended."""
    chosen = [m for m in subset if m in requested]
    return chosen + [m for m in requested if m not in config.PAPER_MODELS]


def tables_dir(args) -> Path:
    path = Path(args.results_dir) / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(args) -> Path:
    path = Path(args.results_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_table(df: pd.DataFrame, path: Path, index: bool = True, float_format: str | None = None) -> Path:
    """Write a CSV table and report its path."""
    df.to_csv(path, index=index, float_format=float_format)
    print(f"wrote {path}")
    return path


def report_figures(paths) -> None:
    for path in paths:
        print(f"wrote {path}")
