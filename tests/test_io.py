import pandas as pd
import pytest

from balportrait import config, io


def test_metadata_covers_the_balanced_dataset():
    metadata = io.load_metadata()
    assert len(metadata) == config.N_PORTRAITS
    assert metadata["ega"].nunique() == 24
    assert (metadata.groupby("ega").size() == 375).all()
    assert set(config.CONTROL_ATTRIBUTES) <= set(metadata.columns)


@pytest.mark.parametrize("scenario,n_items", [("occupations", 100), ("brands", 50), ("activities", 50), ("banking", 50)])
def test_prestige_tables(scenario, n_items):
    prestige = io.load_prestige(scenario)
    assert len(prestige) == n_items
    assert list(prestige.index) == list(range(1, n_items + 1))


def test_released_outputs_are_discovered():
    models = io.discover_models(config.DEFAULT_OUTPUTS_DIR)
    assert list(models) == list(config.PAPER_MODELS)
    assert all(scenarios == list(config.SCENARIOS) for scenarios in models.values())


def test_validation_flags_missing_items_and_extra_rows():
    hashes = io.load_demographics().index[:3]
    raw = pd.DataFrame({"HASH": list(hashes) + ["not_a_portrait"], **{str(i): [5] * 4 for i in range(1, 50)}})
    report = io.validate_model_output(raw, "brands", "toy", hashes)
    assert report.missing_items == [50]
    assert report.extra_hashes == ["not_a_portrait"]
    assert report.fatal


def test_parse_raw_answers(tmp_path):
    (tmp_path / "aaaa.txt").write_text("1:7\n2:11\nnot a line\n3: 4\n999:5\n")
    (tmp_path / "bbbb.txt").write_text("Apple:9\nporsche:3\n")
    occupations = io.parse_raw_answers(tmp_path, "occupations").set_index("HASH")
    assert occupations.loc["aaaa", 1] == 7
    assert pd.isna(occupations.loc["aaaa", 2])  # out of the 1-10 range
    assert occupations.loc["aaaa", 3] == 4
    brands = io.parse_raw_answers(tmp_path, "brands").set_index("HASH")
    items = io.load_prestige_table("brands")
    apple = items.index[items["item_name"] == "Apple"][0]
    porsche = items.index[items["item_name"] == "Porsche"][0]
    assert brands.loc["bbbb", apple] == 9 and brands.loc["bbbb", porsche] == 3
