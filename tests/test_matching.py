import pandas as pd
import pytest

from acbm.matching import MatcherExact, match_psm  # noqa: F401


@pytest.fixture
def setup_data():
    df_pop = pd.DataFrame(
        {
            "hid_pop": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "num_adults_pop": [2, 2, 1, 3, 1, 2, 2, 1, 2, 3],
            "num_children_pop": [1, 0, 2, 1, 0, 1, 0, 2, 1, 1],
            "num_cars_pop": [1, 2, 1, 0, 1, 2, 1, 0, 1, 2],
            "urban_rural_pop": [
                "urban",
                "rural",
                "urban",
                "urban",
                "rural",
                "urban",
                "urban",
                "rural",
                "urban",
                "urban",
            ],
        }
    )

    df_sample = pd.DataFrame(
        {
            "hid_sample": [101, 102, 103, 104, 105],
            "num_adults_sample": [2, 1, 2, 3, 2],
            "num_children_sample": [1, 0, 0, 1, 1],
            "num_cars_sample": [1, 2, 1, 0, 1],
            "urban_rural_sample": ["urban", "rural", "urban", "urban", "rural"],
        }
    )

    matching_dict = {
        "num_adults": ["num_adults_pop", "num_adults_sample"],
        "num_children": ["num_children_pop", "num_children_sample"],
        "num_cars": ["num_cars_pop", "num_cars_sample"],
        "urban_rural": ["urban_rural_pop", "urban_rural_sample"],
    }
    fixed_cols = [
        "num_adults",
        "num_children",
    ]
    optional_cols = [
        "num_cars",
        "urban_rural",
    ]

    return MatcherExact(
        df_pop=df_pop,
        df_pop_id="hid_pop",
        df_sample=df_sample,
        df_sample_id="hid_sample",
        matching_dict=matching_dict,
        fixed_cols=fixed_cols,
        optional_cols=optional_cols,
        n_matches=2,
        chunk_size=2,
        show_progress=False,
    )


def test_post_init(setup_data):
    matcher = setup_data
    assert matcher.fixed_pop_cols == ["num_adults_pop", "num_children_pop"]
    assert matcher.fixed_sample_cols == ["num_adults_sample", "num_children_sample"]
    assert matcher.optional_pop_cols == ["num_cars_pop", "urban_rural_pop"]
    assert matcher.optional_sample_cols == ["num_cars_sample", "urban_rural_sample"]
    assert not matcher.remaining_df_pop.empty
    assert not matcher.remaining_df_sample.empty


@pytest.mark.skip(reason="todo")
def test_match_categorical():
    pass


def test_iterative_match_categorical(setup_data):
    matcher = setup_data
    result = matcher.iterative_match_categorical()
    expected_result = {
        1: [101.0, 105.0],
        2: [103.0],
        3: [],
        4: [104.0],
        5: [102],
        6: [101, 105],
        7: [103.0],
        8: [],
        9: [101.0, 105.0],
        10: [104.0],
    }
    assert result == expected_result


@pytest.mark.skip(reason="todo")
def test_match_individuals():
    pass


@pytest.mark.skip(reason="todo")
def test_match_psm():
    pass
