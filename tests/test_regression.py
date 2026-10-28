import numpy as np
import pandas as pd

from balportrait import regression


def _toy(rng, n=600):
    df = pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], n),
            "age": rng.choice(["25", "40", "55"], n),
            "earrings": rng.choice(["True", "False"], n, p=[0.3, 0.7]),
            "hair_color": rng.choice(["black_hair", "brown_hair", "None"], n, p=[0.6, 0.3, 0.1]),
        }
    )
    df["score"] = (
        0.2
        - 0.05 * (df["gender"] == "Female")
        + 0.03 * (df["age"] == "55")
        + 0.02 * (df["hair_color"] == "brown_hair")
        + rng.normal(0, 0.005, n)
    )
    return df


def test_reference_levels(rng):
    df = _toy(rng)
    X, refs = regression.build_design_matrix(df, ["gender", "age", "earrings", "hair_color"])
    assert refs == {"gender": "Male", "age": "40", "earrings": "False", "hair_color": "Black_Hair"}
    assert "gender_Male" not in X and "earrings_False" not in X and "hair_color_Black_Hair" not in X
    assert "hair_color_None" in X and "const" in X


def test_estimators_recover_effects(rng):
    df = _toy(rng)
    for estimator in ("rlm", "ols"):
        fit = regression.fit_marginal_effects(df, "score", ["gender", "age", "hair_color"], estimator=estimator,
                                              drop_bottom_quantile=None)
        coefs = fit.coefficients().set_index("term")
        assert np.isclose(coefs.loc["gender_Female", "estimate"], -0.05, atol=0.002)
        assert np.isclose(coefs.loc["age_55", "estimate"], 0.03, atol=0.002)
        assert np.isclose(coefs.loc["hair_color_Brown_Hair", "estimate"], 0.02, atol=0.002)
        assert coefs.loc["age_25", "p_value"] > 0.001


def test_bottom_quantile_filter(rng):
    df = _toy(rng)
    fit = regression.fit_marginal_effects(df, "score", ["gender"], drop_bottom_quantile=0.1)
    assert fit.n_obs == int((df["score"] > df["score"].quantile(0.1)).sum())


def test_split_term():
    attributes = ("hair_color", "hair_coverage", "age")
    assert regression.split_term("hair_color_Black_Hair", attributes) == ("hair_color", "Black_Hair")
    assert regression.split_term("age_25", attributes) == ("age", "25")


def test_vif_of_independent_dummies_is_close_to_one(rng):
    df = _toy(rng, n=3000)
    X, _ = regression.build_design_matrix(df, ["gender", "earrings"])
    assert (regression.vif_scores(X) < 1.05).all()
