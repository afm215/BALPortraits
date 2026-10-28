"""Constants shared by the whole package: paths, scenarios, models, display settings."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
PRESTIGE_DIR = METADATA_DIR / "prestige"
PROMPTS_DIR = DATA_DIR / "prompts"
DEFAULT_OUTPUTS_DIR = DATA_DIR / "model_outputs"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

# ── Scenarios ────────────────────────────────────────────────────────────────
SCENARIOS = ("occupations", "brands", "activities", "banking")
SCENARIO_LABELS = {
    "global": "Global",
    "occupations": "Occupations",
    "brands": "Brands",
    "activities": "Activities",
    "banking": "Banking",
}

# ── Demographic segments ─────────────────────────────────────────────────────
ETHNICITIES = ("African", "Asian", "Caucasian", "Indian")
GENDERS = ("Female", "Male")
AGES = ("25", "40", "55")
N_PORTRAITS = 9000

DEMOGRAPHIC_ATTRIBUTES = ("ethnicity", "gender", "age")

# Appearance categories used as control variables in the marginal-effect analysis.
CONTROL_ATTRIBUTES = (
    "bindi",
    "earrings",
    "eyeglasses",
    "necklace",
    "beard_style",
    "chin_shape",
    "eye_color",
    "hair_color",
    "hair_coverage",
    "hair_length",
    "hairstyle_texture_or_arrangement",
    "headwear_type",
    "moustache_style",
)
ALL_ATTRIBUTES = DEMOGRAPHIC_ATTRIBUTES + CONTROL_ATTRIBUTES

# ── Audited models ───────────────────────────────────────────────────────────
PAPER_MODELS = (
    "gpt54mini",
    "claudehaiku45",
    "gemini31flashlite",
    "gemma4_4B",
    "gemma4_26B",
    "gemma4_31B",
    "InternVL3_5-4B",
    "InternVL3_5-8B",
    "InternVL3_5-30B-A3B",
    "ministral3B",
    "ministral8B",
    "ministral14B",
    "QWEN_3_4B",
    "QWEN_3_8B",
    "QWEN_3_32B",
    "Qwen3-VL-235B-A22B-Instruct",
)
DISPLAY_NAMES = {
    "gpt54mini": "GPT54M",
    "claudehaiku45": "CLAH45",
    "gemini31flashlite": "GEM31F",
    "gemma4_4B": "GM4-4B",
    "gemma4_26B": "GM4-26B",
    "gemma4_31B": "GM4-31B",
    "InternVL3_5-4B": "IN3-4B",
    "InternVL3_5-8B": "IN3-8B",
    "InternVL3_5-30B-A3B": "IN3-30B",
    "ministral3B": "MI3-3B",
    "ministral8B": "MI3-8B",
    "ministral14B": "MI3-14B",
    "QWEN_3_4B": "QW3-4B",
    "QWEN_3_8B": "QW3-8B",
    "QWEN_3_32B": "QW3-32B",
    "Qwen3-VL-235B-A22B-Instruct": "QW3-235B",
}

# Model subsets shown in the paper figures.
SELECTED_MODELS = (
    "gpt54mini",
    "claudehaiku45",
    "gemini31flashlite",
    "gemma4_31B",
    "InternVL3_5-30B-A3B",
    "ministral14B",
    "Qwen3-VL-235B-A22B-Instruct",
)
FRONTIER_MODELS = ("gpt54mini", "claudehaiku45", "gemini31flashlite")
SIGNIFICANCE_MODELS = ("gpt54mini", "claudehaiku45")


def display_name(model: str) -> str:
    """Short display name of a model; unknown (user) models keep their identifier."""
    return DISPLAY_NAMES.get(model, model)


def order_models(models):
    """Order models as in the paper, followed by any other model in alphabetical order."""
    models = list(dict.fromkeys(models))
    known = [m for m in PAPER_MODELS if m in models]
    return known + sorted(m for m in models if m not in PAPER_MODELS)


# ── Colours ──────────────────────────────────────────────────────────────────
SCENARIO_COLORS = {
    "global": "#4C78A8",
    "occupations": "#F58518",
    "brands": "#54A24B",
    "activities": "#B279A2",
    "banking": "#E45756",
}

# ── Marginal-effect analysis ─────────────────────────────────────────────────
DEMOGRAPHIC_REFERENCES = {"ethnicity": "Caucasian", "gender": "Male", "age": "40"}
ME_DROP_BOTTOM_QUANTILE = 0.01
ME_P_ADJUST = "none"
ME_ALPHA = 0.05

# Attribute groups of the face-attribute marginal-effect figures (file stem → categories).
ATTRIBUTE_FIGURE_GROUPS = {
    "binary_attribute": ("moustache_style", "eyeglasses", "earrings", "bindi", "necklace"),
    "headwear_type": ("headwear_type",),
    "hair_color": ("hair_color",),
    "eye_color": ("eye_color",),
    "hair_coverage_length": ("hair_coverage", "hair_length"),
    "chin_beard": ("chin_shape", "beard_style"),
}

# ── Significance summaries ───────────────────────────────────────────────────
# Bins of the Mann-Whitney U pies of the degree-of-bias figure (lower bound inclusive).
P_VALUE_BINS = (
    ("p < 0.001", 0.0, 0.001),
    ("0.01 > p ≥ 0.001", 0.001, 0.01),
    ("0.05 > p ≥ 0.01", 0.01, 0.05),
    ("p ≥ 0.05", 0.05, 1.0 + 1e-12),
)
