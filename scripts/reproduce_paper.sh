#!/usr/bin/env bash
# Regenerate every table and figure of the paper into results/ (about 5 minutes).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

"$PYTHON" scripts/compute_scores.py --quiet
"$PYTHON" scripts/degree_of_bias.py --quiet
"$PYTHON" scripts/group_ranks.py --quiet
"$PYTHON" scripts/significance_tests.py --quiet --latex
"$PYTHON" scripts/marginal_effects.py --quiet --diagnostics
"$PYTHON" scripts/dataset_attributes.py --quiet
"$PYTHON" scripts/fitness_dispersion.py --quiet
