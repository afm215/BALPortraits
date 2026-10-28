"""Loading and validation of the released metadata and of model outputs."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ── Metadata ─────────────────────────────────────────────────────────────────
def load_demographics(path: Path | None = None) -> pd.DataFrame:
    """Load the demographic labels of the 9,000 portraits.

    Returns
    -------
    pd.DataFrame
        Indexed by ``HASH`` with string columns ``ethnicity``, ``gender``, ``age``
        and ``ega`` (e.g. ``"African|Female|25"``).
    """
    path = path or config.METADATA_DIR / "demographics.csv"
    df = pd.read_csv(path, dtype=str).set_index("HASH")
    df.columns = [c.lower() for c in df.columns]
    df["ega"] = df["ethnicity"] + "|" + df["gender"] + "|" + df["age"]
    return df


def load_face_attributes(path: Path | None = None) -> pd.DataFrame:
    """Load the 13 appearance categories (strings; ``"None"`` marks an absent value)."""
    path = path or config.METADATA_DIR / "face_attributes.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index("HASH")


def load_metadata() -> pd.DataFrame:
    """Demographics joined with face attributes, one row per portrait."""
    demographics = load_demographics()
    attributes = load_face_attributes()
    if set(demographics.index) != set(attributes.index):
        raise ValueError("demographics.csv and face_attributes.csv do not cover the same portraits")
    return demographics.join(attributes, how="left")


def load_prestige_table(scenario: str) -> pd.DataFrame:
    """Item list of a scenario: ``item_id`` (index), ``item_name``, ``item_category``, ``prestige``."""
    path = config.PRESTIGE_DIR / f"{scenario}.tsv"
    return pd.read_csv(path, sep="\t").set_index("item_id")


def load_prestige(scenario: str) -> pd.Series:
    """Raw prestige score of every item of a scenario, indexed by ``item_id``."""
    return load_prestige_table(scenario)["prestige"]


# ── Model outputs ────────────────────────────────────────────────────────────
def model_output_path(outputs_dir: Path, scenario: str, model: str) -> Path | None:
    """Path of ``<outputs_dir>/<scenario>/<model>.csv[.gz]``, or None if absent."""
    for suffix in (".csv.gz", ".csv"):
        path = Path(outputs_dir) / scenario / f"{model}{suffix}"
        if path.exists():
            return path
    return None


def discover_models(outputs_dir: Path) -> dict[str, list[str]]:
    """Map every model found in ``outputs_dir`` to the scenarios it has outputs for."""
    found: dict[str, list[str]] = {}
    for scenario in config.SCENARIOS:
        scenario_dir = Path(outputs_dir) / scenario
        if not scenario_dir.is_dir():
            continue
        for path in sorted(scenario_dir.iterdir()):
            name = path.name
            for suffix in (".csv.gz", ".csv"):
                if name.endswith(suffix):
                    found.setdefault(name[: -len(suffix)], []).append(scenario)
                    break
    return {model: found[model] for model in config.order_models(found)}


@dataclass
class ValidationReport:
    """Result of checking one model output against the dataset and the item list."""

    scenario: str
    model: str
    n_rows: int
    missing_hashes: list[str] = field(default_factory=list)
    extra_hashes: list[str] = field(default_factory=list)
    duplicated_hashes: list[str] = field(default_factory=list)
    missing_items: list[int] = field(default_factory=list)
    extra_items: list[str] = field(default_factory=list)
    n_nan_cells: int = 0
    all_nan_hashes: list[str] = field(default_factory=list)

    @property
    def fatal(self) -> bool:
        return bool(self.missing_items or self.extra_items or self.duplicated_hashes)

    def summary(self) -> str:
        lines = [f"[{self.model} / {self.scenario}] {self.n_rows} rows"]
        if self.missing_hashes:
            lines.append(f"  {len(self.missing_hashes)} dataset portraits have no output")
        if self.extra_hashes:
            lines.append(f"  {len(self.extra_hashes)} rows are not BALPortrait portraits (dropped)")
        if self.duplicated_hashes:
            lines.append(f"  ERROR: {len(self.duplicated_hashes)} duplicated HASH values")
        if self.missing_items:
            lines.append(f"  ERROR: missing item columns {self.missing_items}")
        if self.extra_items:
            lines.append(f"  ERROR: unknown item columns {self.extra_items}")
        if self.n_nan_cells:
            lines.append(f"  {self.n_nan_cells} missing scores (ignored in the portrait mean)")
        if self.all_nan_hashes:
            lines.append(f"  {len(self.all_nan_hashes)} portraits have no valid score (dropped)")
        return "\n".join(lines)

    def raise_if_fatal(self) -> None:
        if self.fatal:
            raise ValueError(self.summary())


def validate_model_output(raw: pd.DataFrame, scenario: str, model: str, hashes) -> ValidationReport:
    """Check a raw output table (``HASH`` column + one column per item id)."""
    if "HASH" not in raw.columns:
        raise ValueError(f"[{model} / {scenario}] the output file has no HASH column")
    hashes = pd.Index(hashes)
    item_ids = set(load_prestige(scenario).index)
    columns = [c for c in raw.columns if c != "HASH"]
    parsed = {c: int(c) for c in columns if re.fullmatch(r"\s*\d+\s*", str(c))}

    report = ValidationReport(scenario=scenario, model=model, n_rows=len(raw))
    report.duplicated_hashes = sorted(raw.loc[raw["HASH"].duplicated(), "HASH"].unique())
    report.missing_hashes = sorted(set(hashes) - set(raw["HASH"]))
    report.extra_hashes = sorted(set(raw["HASH"]) - set(hashes))
    report.missing_items = sorted(item_ids - set(parsed.values()))
    report.extra_items = sorted(str(c) for c in columns if c not in parsed or parsed[c] not in item_ids)

    values = raw[columns].apply(pd.to_numeric, errors="coerce")
    in_dataset = raw["HASH"].isin(hashes).to_numpy()
    report.n_nan_cells = int(values[in_dataset].isna().sum().sum())
    all_nan = values.isna().all(axis=1).to_numpy() & in_dataset
    report.all_nan_hashes = sorted(raw.loc[all_nan, "HASH"])
    return report


def load_model_output(outputs_dir: Path, scenario: str, model: str, hashes, verbose: bool = True) -> pd.DataFrame:
    """Load the fitness scores of one model for one scenario.

    Rows outside ``hashes`` (the dataset portraits) are dropped, as are portraits
    without any valid score. Non-numeric cells become NaN.

    Returns
    -------
    pd.DataFrame
        Indexed by ``HASH``, one float column per ``item_id`` (int), sorted by id.
    """
    path = model_output_path(outputs_dir, scenario, model)
    if path is None:
        raise FileNotFoundError(f"No output for model '{model}' and scenario '{scenario}' in {outputs_dir}")
    raw = pd.read_csv(path, dtype={"HASH": str})
    report = validate_model_output(raw, scenario, model, hashes)
    report.raise_if_fatal()
    if verbose and (report.missing_hashes or report.extra_hashes or report.n_nan_cells):
        print(report.summary())

    df = raw.set_index("HASH")
    df.columns = [int(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    df = df.loc[df.index.isin(pd.Index(hashes))]
    df = df.loc[~df.isna().all(axis=1)]
    return df[sorted(df.columns)]


def parse_raw_answers(raw_dir: Path, scenario: str, score_range: tuple[float, float] = (1, 10)) -> pd.DataFrame:
    """Build an output table from raw per-image answers.

    ``raw_dir`` contains one ``<HASH>.txt`` file per portrait whose lines follow the
    prompt format ``ITEM:SCORE``. ``ITEM`` is the numeric item id or, as in the brands
    prompt, the item name (matched case-insensitively). Lines that do not parse, refer
    to an unknown item or have a score outside ``score_range`` are ignored; items
    without a valid answer are left empty.

    Returns
    -------
    pd.DataFrame
        ``HASH`` column followed by one column per item id.
    """
    items = load_prestige_table(scenario)
    by_name = {str(name).strip().casefold(): item_id for item_id, name in items["item_name"].items()}
    valid_ids = set(items.index)
    low, high = score_range

    rows = []
    for path in sorted(Path(raw_dir).glob("*.txt")):
        row = {"HASH": path.stem}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.rsplit(":", 1))
            item_id = int(key) if key.isdigit() and int(key) in valid_ids else by_name.get(key.casefold())
            try:
                score = float(value)
            except ValueError:
                continue
            if item_id is None or not (low <= score <= high):
                continue
            row.setdefault(item_id, score)
        rows.append(row)
    if not rows:
        warnings.warn(f"No .txt answer file found in {raw_dir}")
    table = pd.DataFrame(rows, columns=["HASH", *sorted(valid_ids)])
    return table.astype({item_id: float for item_id in valid_ids})


def write_model_output(table: pd.DataFrame, path: Path) -> None:
    """Write an output table, keeping integer scores as integers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = table.copy()
    scores = out.drop(columns="HASH")
    if np.all(np.isnan(scores.to_numpy()) | (scores.to_numpy() == np.round(scores.to_numpy()))):
        out[scores.columns] = scores.astype("Int64")
    out.to_csv(path, index=False)
