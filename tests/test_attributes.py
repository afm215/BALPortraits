import numpy as np
import pandas as pd

from balportrait import attributes


def test_cramers_v_bounds(rng):
    x = rng.choice(list("abc"), 5000)
    independent = rng.choice(list("xyz"), 5000)
    assert attributes.cramers_v_corrected(x, independent) < 0.03
    assert np.isclose(attributes.cramers_v_corrected(x, x), 1.0, atol=1e-3)


def test_phi_of_a_binary_variable_with_itself():
    df = pd.DataFrame({"a": ["True", "False", "True", "False"], "b": ["x", "y", "x", "y"]})
    phi = attributes.phi_matrix(df, "a", "b")
    assert np.isclose(phi.loc["True", "x"], 1.0) and np.isclose(phi.loc["True", "y"], -1.0)


def test_counts_exclude_none():
    df = pd.DataFrame({"headwear_type": ["None", "hood", "hood", "hijab"]})
    counts = attributes.attribute_counts(df, ["headwear_type"])
    assert counts["value"].tolist() == ["hood", "hijab"]
    assert counts["count"].tolist() == [2, 1]
