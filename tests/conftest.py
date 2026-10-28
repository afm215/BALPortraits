import numpy as np
import pandas as pd
import pytest

from balportrait import config


@pytest.fixture
def toy_ranks():
    """24 EGA segments × 10 portraits with ranks 1..240 assigned in segment order."""
    rows = []
    rank = 1
    for e in config.ETHNICITIES:
        for g in config.GENDERS:
            for a in config.AGES:
                for _ in range(10):
                    rows.append({"ethnicity": e, "gender": g, "age": a, "ega": f"{e}|{g}|{a}", "rank": float(rank)})
                    rank += 1
    return pd.DataFrame(rows)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
