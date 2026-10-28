# BALPortraits: Probing Implicit Demographic Bias in VLMs

Code and data for the WACV 2027 paper *Portrait Fitness as a Pretext: Probing Implicit Demographic Bias in VLMs*.

BALPortraits is a balanced dataset of 9,000 fictional portraits spread over 24 ethnicity × gender × age (EGA) segments: 4 ethnic segments (African, Asian, Caucasian, Indian), 2 genders and 3 age anchors (25, 40, 55), with 375 portraits per segment. Vision-language models (VLMs) score how well each portrait fits every item of four advertising-casting scenarios: occupations (100 items), brands, activities, and banking or insurance products (50 items each). Item-level scores are aggregated with scenario-specific prestige scores into a portrait score. Portraits are then ranked, and the ranks are compared across demographic segments.

This repository contains:
- the demographic and face-attribute metadata of the portraits;
- the item lists with their prestige scores, and the scenario prompts;
- the fitness scores of the 16 audited VLMs;
- the code that computes every analysis of the paper, and that runs the same audit on the outputs of any other model.

## Installation

```bash
git clone <repository-url> && cd BALPortrait
pip install -e .            # or: pip install -r requirements.txt (exact versions used for the paper)
```

Python ≥ 3.9. Dependencies: numpy, pandas, scipy, statsmodels, matplotlib.

## Repository structure

```
data/
  metadata/demographics.csv        HASH, ETHNICITY, GENDER, AGE of the 9,000 portraits
  metadata/face_attributes.csv     HASH + 13 appearance categories
  metadata/prestige/<scenario>.tsv item_id, item_name, item_category, prestige
  prompts/<scenario>.txt           fitness-scoring prompt of each scenario
  model_outputs/<scenario>/<model>.csv.gz   fitness scores of the 16 audited VLMs
balportrait/                       Python package (all computations)
  io.py            loading and validation of metadata and model outputs
  scoring.py       portrait prestige scores and ranks
  bias_metrics.py  group mean ranks, degree of bias, null baseline
  significance.py  pairwise Mann-Whitney U tests, rank-biserial correlation
  regression.py    marginal-effect regressions, VIF
  attributes.py    attribute counts, Cramér's V, phi coefficients
  plotting/        figure functions
scripts/                           command-line entry points (one per analysis)
tests/                             unit tests and checks against the paper
```

All scripts write to `results/tables/` and `results/figures/` by default (`--results-dir` to change it).

## Data

**Portraits.** Images are named `<HASH>.jpg`, where `HASH` is a 16-character hexadecimal identifier. `HASH` is the key that links images, metadata and model outputs. The image dataset can be found under this link https://drive.google.com/drive/folders/1A1Z6lHlV_MCqxH7A5-_GJPxOsWezwLRw?usp=sharing.

**`demographics.csv`.** The ethnicity, gender and age segment of every portrait. These labels are used to control visual variation in the generated dataset. They should not be interpreted as biological categories or fixed identity groups.

**`face_attributes.csv`.** Visual attributes annotated with GPT-5.4 and used as control variables in the marginal-effect analysis. The 13 categories are bindi, earrings, eyeglasses, necklace, beard style, chin shape, eye color, hair color, hair coverage, hair length, hairstyle texture or arrangement, headwear type and moustache style (49 values in total). Binary accessories are `True`/`False`. `None` means that the attribute is absent or could not be inferred.

**`prestige/<scenario>.tsv`.** One row per item, with its prestige score. Occupation prestige comes from Hughes et al. (2024). For brands, activities and banking/insurance products, prestige is the aggregate of GPT-5.4, Claude Opus 4.7 and DeepSeek-V4-Pro estimates, after min-max normalization.

**`prompts/<scenario>.txt`.** The scoring prompt of each scenario. `{items}` is replaced by the item list, one item per line, in a random order for each image:

| scenario | item line | answer line |
|---|---|---|
| occupations | `ID\|NAME` | `ID:SCORE` |
| brands | `CATEGORY\|NAME` | `NAME:SCORE` |
| activities, banking | `ID\|CATEGORY\|NAME` | `ID:SCORE` |

**`model_outputs/<scenario>/<model>.csv.gz`.** One row per portrait: a `HASH` column, then one column per `item_id` holding the 1–10 fitness score. When portraits have no output for some model and scenario (failed answers), they are left out of the analyses of that model and scenario.

## Method

**Portrait score.** For each model and scenario $S$, fitness scores are min-max normalized per item over the portraits ($\hat f_i$). Item prestige is standardized within the scenario. The portrait score is

$$a_S = \frac{1}{|S|}\sum_{i=1}^{|S|} \frac{p_i-\mu_p}{\sigma_p}\,\hat f_i .$$

Portraits are ranked by $a_S$; rank 1 is the highest score. The global rank of a portrait is the mean of its four scenario ranks.

**Degree of bias (DoB).** The standard deviation of the mean ranks of the 24 EGA segments, per model and scenario. The global DoB of a model is the mean over the four scenarios. The *equity line* is the DoB expected under demographic equity. It is obtained by replacing every fitness score with a random integer between 1 and 10 and keeping the rest of the pipeline unchanged (20 random seeds).

**Significance and effect size.** Groups are compared with two-sided Mann-Whitney U tests on the portrait ranks. The rank-biserial correlation $r_{rb} = 1 - 2U_A/(n_A n_B)$ is reported with every test. Since rank 1 is the best position, $r_{rb} > 0$ means that group A is ranked better than group B. In the degree-of-bias figure, the pies summarize the 276 pairwise tests between EGA segments on the global rank, with Benjamini-Hochberg adjustment over the 276 tests of each model.

**Marginal effects.** The portrait score is regressed on dummy-encoded categorical attributes, using robust linear regression: Huber M-estimator with $k = 1.345$, fitted by iteratively reweighted least squares (statsmodels `RLM`), with asymptotic Wald z-tests. Each coefficient is the expected change of the score when an attribute switches from its reference level to the given level, all other attributes being equal.
- References are Caucasian, Male and 40 for the demographic attributes. For appearance attributes, the reference is `False` for binary accessories and the most frequent level otherwise.
- The *controlled* setup includes ethnicity, gender, age and the 13 appearance categories. The *uncontrolled* setup includes ethnicity, gender and age only.
- Error bars are half the width of the 95% confidence interval. In the figures, bars with p > 0.05 are transparent.

**Dataset attributes.** Associations between attributes are measured with the bias-corrected Cramér's V (Bergsma, 2013). For pairs with Ṽ > 0.3, phi coefficients are computed between all their levels. The variance inflation factor (VIF) is computed for every dummy of the marginal-effect design matrix.

## Reproducing the paper

```bash
bash scripts/reproduce_paper.sh      # about 5 minutes; set PYTHON=... to choose the interpreter
```

| Paper | Script | Output (`results/`) |
|---|---|---|
| Degree of bias with equity line and Mann-Whitney U pies (Fig. 2) | `degree_of_bias.py` | `figures/global_experiment_ranks_equity_line`, `tables/degree_of_bias.csv`, `tables/null_baseline.csv`, `tables/global_mwu_summary.csv` |
| One-way mean ranks, selected and other VLMs | `group_ranks.py` | `figures/detailed_experiment_ranks`, `figures/app_detailed_experiment_ranks`, `tables/mean_rank_one_way.csv` |
| Intersectional EGA mean ranks, frontier and other VLMs | `group_ranks.py` | `figures/detailed_frontier_matrices_ranks`, `figures/app_detailed_frontier_matrices_ranks`, `tables/mean_rank_ega.csv` |
| Marginal effects of demographic attributes (main text and appendix) | `marginal_effects.py` | `figures/marginal_effects`, `figures/eth_gen_age_marginal_effects`, `figures/eth_gen_age_marginal_effects_all_models`, `tables/marginal_effects_controlled.csv` |
| Impact of controlling on face attributes | `marginal_effects.py` | `figures/eth_gen_age_marginal_effects_no_control`, `tables/marginal_effects_uncontrolled.csv`, `tables/control_impact.csv` |
| Marginal effects of face attributes | `marginal_effects.py` | `figures/{binary_attribute,chin_beard,hair_coverage_length,eye_color,hair_color,headwear_type}_marginal_effects` |
| OLS residual diagnostics | `marginal_effects.py --diagnostics` | `figures/residual_plot_not_fixed`, `figures/residual_plot_fixed`, `tables/regression_diagnostics.csv` |
| Pairwise Mann-Whitney U p-values and rank-biserial correlations (GPT54M, CLAH45) | `significance_tests.py` | `figures/gpt54min_claudehaiku_{p_value,r_rb}`, `figures/mwu_joint_ega_{p_value,r_rb}`, `tables/pairwise_mwu.csv` |
| Mean rank-biserial correlation vs degree of bias | `significance_tests.py` | `tables/mean_abs_rrb_ega.csv`, `tables/rrb_vs_dob_per_scenario.csv`, `tables/rrb_vs_dob_agreement.csv` |
| List of attributes with counts | `dataset_attributes.py` | `tables/attribute_counts.{csv,tex}` |
| Cramér's V, phi coefficients, VIF | `dataset_attributes.py` | `figures/cramers_v_matrix_corrected`, `figures/phi_correlations_aggregated`, `figures/vif_scores`, `tables/cramers_v.csv`, `tables/phi_pairs.csv`, `tables/vif.csv` |
| Standard deviation of the portrait scores | `fitness_dispersion.py` | `figures/overall_prestige_std_by_model_experience`, `tables/score_dispersion.csv` |
| Portrait scores and ranks | `compute_scores.py` | `scores/<model>.csv` |

Figures are written as PDF and PNG (`--formats` to change).

## Auditing your own model

1. **Run your VLM** on the 9,000 portraits with the four prompts of `data/prompts/`. Shuffle the item list for each image, and save the raw answer of each portrait as `<HASH>.txt`, in one folder per scenario.
2. **Convert the answers** into the model-output format, once per scenario:
   ```bash
   python scripts/prepare_model_outputs.py --raw-dir answers/occupations --scenario occupations --model my_vlm
   ```
   This writes `data/model_outputs/occupations/my_vlm.csv.gz` and reports missing portraits and invalid answers. If your outputs are already tables, write them directly in the format of the *Data* section: `HASH` column plus one column per `item_id`, in `<outputs-dir>/<scenario>/<model>.csv[.gz]`.
3. **Run the audit**:
   ```bash
   python scripts/audit_model.py --model my_vlm                      # outputs in data/model_outputs
   python scripts/audit_model.py --model my_vlm --outputs-dir my_dir # outputs elsewhere
   ```
   Results go to `results/audit_my_vlm/`:
   - the degree of bias next to the 16 audited VLMs (`--compare-with none` to omit them), with the equity line and significance pies;
   - one-way and intersectional mean ranks;
   - all pairwise tests with rank-biserial correlations;
   - controlled and uncontrolled marginal effects.

Every analysis script also accepts `--outputs-dir` and `--models`, so the paper figures can be regenerated with additional models. Models outside the paper set are added to the main figures.

## Script options

Common options: `--outputs-dir`, `--models`, `--results-dir`, `--formats`, `--quiet`.

| Script | Specific options |
|---|---|
| `degree_of_bias.py` | `--null-seeds` (20), `--seed` (0), `--p-adjust` of the pies (`fdr_bh`) |
| `significance_tests.py` | `--p-adjust` (`none`; `fdr_bh` adds BH-adjusted p-values per model, scenario and attribute in column `p_adj`), `--heatmap-models`, `--latex` |
| `marginal_effects.py` | `--estimator` (`rlm` or `ols`), `--drop-bottom-quantile` (0.01: portraits whose score is not above the 1% quantile of their model and scenario are excluded before fitting; 0 keeps all portraits), `--p-adjust` (`none` or `fdr_bh`, within each fit), `--alpha` (0.05), `--diagnostics` |
| `dataset_attributes.py` | `--phi-threshold` (0.3), `--phi-exclude` (`eye_color`), `--vif-model` / `--vif-scenario` (design matrix of the GPT54M occupations fit; `--vif-model none` uses all portraits), `--drop-bottom-quantile` |
| `audit_model.py` | `--model` (required), `--compare-with` (`paper` or `none`), `--paper-outputs-dir`, `--null-seeds` |

## Tests

```bash
pytest              # unit tests (a few seconds)
pytest -m slow      # checks of the released data and code against numbers reported in the paper
```

## Citation

```bibtex
@inproceedings{fourniermontgieux2027portrait,
  title     = {Portrait Fitness as a Pretext: Probing Implicit Demographic Bias in VLMs},
  author    = {Fournier-Montgieux, Alexandre and Elbouanani, Akram and Tuo, Aboubacar and
               Popescu, Adrian and Le Borgne, Herv{\'e} and Luvison, Bertrand},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Acknowledgements

This work was partly supported by the SHARP ANR project ANR-23-PEIA-0008 in the context of the France 2030 program. It was made possible by the use of the FactoryIA supercomputer, financially supported by the Ile-De-France Regional Council.
