"""Checks of the released code and data against numbers reported in the paper (slow)."""

import numpy as np
import pandas as pd
import pytest

from balportrait import attributes, bias_metrics, config, io, regression, scoring, significance

pytestmark = pytest.mark.slow

CONTROLLED = config.CONTROL_ATTRIBUTES + config.DEMOGRAPHIC_ATTRIBUTES


@pytest.fixture(scope="module")
def metadata():
    return io.load_metadata()


@pytest.fixture(scope="module")
def all_scores(metadata):
    return scoring.compute_all_scores(config.DEFAULT_OUTPUTS_DIR, config.PAPER_MODELS, metadata, verbose=False)


def test_attribute_table(metadata):
    counts = attributes.attribute_counts(metadata).set_index(["category", "value"])["count"]
    expected = {
        ("hair_coverage", "thinning_hair"): 1272, ("hair_coverage", "bald"): 647,
        ("headwear_type", "hood"): 197, ("headwear_type", "yarmulke"): 12,
        ("eye_color", "brown_eyes"): 8046, ("hair_color", "black_hair"): 5320,
        ("chin_shape", "round_chin"): 6249, ("bindi", "True"): 445, ("necklace", "True"): 40,
        ("hairstyle_texture_or_arrangement", "side_part"): 2433, ("moustache_style", "mustache"): 238,
    }
    for key, value in expected.items():
        assert counts[key] == value
    controls = counts.reset_index()
    controls = controls[~controls["category"].isin(config.DEMOGRAPHIC_ATTRIBUTES)]
    assert controls["category"].nunique() == 13 and len(controls) == 49


def test_cramers_v_and_phi(metadata):
    v = attributes.cramers_v_matrix(metadata)
    assert np.isclose(v.loc["hair_length", "gender"], 0.569, atol=0.002)
    assert np.isclose(v.loc["hair_color", "ethnicity"], 0.428, atol=0.002)
    assert v.loc["ethnicity", "gender"] < 0.005 and v.loc["ethnicity", "age"] < 0.005
    assert len(attributes.strong_pairs(v, 0.3, exclude=("eye_color",))) == 9
    assert np.isclose(attributes.phi_matrix(metadata, "bindi", "ethnicity").loc["True", "Indian"], 0.39, atol=0.01)
    assert np.isclose(attributes.phi_matrix(metadata, "hair_length", "gender").loc["short_hair", "Male"], 0.52, atol=0.01)
    assert np.isclose(attributes.phi_matrix(metadata, "hair_color", "hair_length").loc["None", "None"], 0.69, atol=0.01)


def test_vif_below_threshold(all_scores):
    fit = regression.fit_marginal_effects(all_scores["gpt54mini"], "score_occupations", CONTROLLED)
    vif = regression.vif_scores(fit.X)
    assert np.isclose(vif.max(), 3.243, atol=0.01) and vif.idxmax() == "ethnicity_African"


def test_equity_line(metadata):
    null = bias_metrics.null_baseline(metadata, n_seeds=20)
    assert 120 < null["dob"].mean() < 140


@pytest.mark.parametrize(
    "model,values",
    [
        ("claudehaiku45", (690, 980, 1260, 1170)),
        ("gemma4_26B", (690, 1025, 1150, 1295)),
        ("QWEN_3_4B", (295, 655, 1005, 410)),
        ("Qwen3-VL-235B-A22B-Instruct", (655, 720, 645, 665)),
    ],
)
def test_degree_of_bias_against_figure(all_scores, model, values):
    dob = bias_metrics.dob_table({model: all_scores[model]})
    for scenario, value in zip(config.SCENARIOS, values):
        assert abs(dob.loc[model, scenario] - value) / value < 0.05


def test_rank_biserial_and_dob_orderings_agree(all_scores):
    pairwise = significance.pairwise_tables(all_scores, attributes=("ega",))
    _, agreement = significance.ranking_agreement(
        significance.mean_abs_rrb_table(pairwise), bias_metrics.dob_table(all_scores)
    )
    assert np.isclose(agreement["spearman_rho"].mean(), 0.994, atol=0.01)


def test_control_reverses_gemma_african_banking_effect(all_scores):
    scores = all_scores["gemma4_31B"]
    controlled = regression.fit_marginal_effects(scores, "score_banking", CONTROLLED).coefficients()
    uncontrolled = regression.fit_marginal_effects(scores, "score_banking", config.DEMOGRAPHIC_ATTRIBUTES).coefficients()
    c = controlled.set_index("term").loc["ethnicity_African"]
    u = uncontrolled.set_index("term").loc["ethnicity_African"]
    assert np.isclose(c["estimate"], 0.021, atol=0.002) and c["p_value"] < 0.001
    assert np.isclose(u["estimate"], -0.008, atol=0.002) and u["p_value"] < 0.01
    assert isinstance(controlled, pd.DataFrame)
