import pandas as pd
import pytest

from acbm.matching import (
    MatcherExact,
    match_individuals,
    match_remaining_individuals,
    matched_ids_from_right_for_left,
)


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


def get_test_dfs_and_mapping(mapping):
    samples = [
        [0, 1, 3, 1, 0],
        [1, 2, 0, 2, 1],
        [2, 2, 1, 1, 2],
        [3, 2, 2, 2, 3],
        [4, 3, 0, 1, 4],
        [5, 3, 1, 2, 5],
        [6, 3, 3, 1, 6],
        [7, 4, 1, 2, 7],
        [8, 4, 1, 1, 8],
        [9, 4, 3, 2, 9],
    ]
    columns = ["id", "hid", "age_group", "sex", "eco_stat_0"]
    return (
        pd.DataFrame(
            samples,
            columns=columns,
        ),
        pd.DataFrame(
            [
                # Use eneumerate for ids, keep hids in same order
                [id, samples[id][1], *samples[idx][2:]]
                for id, idx in enumerate(mapping)
            ],
            columns=columns,
        ),
        mapping,
    )


@pytest.fixture
def ind_hh():
    mapping_ind_within_hh = [0, 3, 1, 2, 4, 6, 5, 9, 8, 7]
    return get_test_dfs_and_mapping(mapping_ind_within_hh)


@pytest.fixture
def ind_rem():
    mapping_ind = [1, 7, 9, 3, 0, 5, 2, 6, 8, 4]
    return get_test_dfs_and_mapping(mapping_ind)


def check_matches(df1, mapping, matches_dict, remaining_ids=None):
    """Checks that the expected indices are present in the returned matches dict"""
    expected = {
        k: v
        for k, v in {mapping[i]: i for i in range(df1.shape[0])}.items()
        if k < (len(remaining_ids) if remaining_ids is not None else df1.shape[0])
    }
    # Check equal after sorting by key
    assert dict(sorted(matches_dict.items())) == dict(sorted(expected.items()))


def check_equals(df1, df2, matches_dict, expect_hid_equals=True):
    """Checks that series are equal after merging the two dataframes with matches"""
    df1["matched_id"] = matched_ids_from_right_for_left(
        df1, df2, matches_dict, right_id="id"
    )
    df = (
        df1.merge(df2, left_on="matched_id", right_on="id", how="left")
        .dropna()
        .astype(int)
    )
    if expect_hid_equals:
        assert df["hid_x"].equals(df["hid_y"])
    assert df["age_group_x"].equals(df["age_group_y"])
    assert df["sex_x"].equals(df["sex_y"])
    assert df["eco_stat_0_x"].equals(df["eco_stat_0_y"])


def test_individual_matching(ind_hh):
    df1, df2, mapping = ind_hh
    matches_dict = match_individuals(
        df1,
        df2,
        df1_id="hid",
        df2_id="hid",
        matching_columns=["age_group", "sex"],
        matches_hh={i: i for i in df1["hid"].unique()},
    )
    check_matches(df1, mapping, matches_dict, remaining_ids=None)
    check_equals(df1, df2, matches_dict)


def test_remaining_individual_matching(ind_rem):
    df1, df2, mapping = ind_rem
    remaining_ids = [0, 1, 2, 3, 4, 5, 6]
    matches_dict = match_remaining_individuals(
        df1,
        df2,
        # A sample of remaining_ids
        remaining_ids=remaining_ids,
        matching_columns=["age_group", "sex", "eco_stat_0"],
    )
    check_matches(df1, mapping, matches_dict, remaining_ids)
    check_equals(df1, df2, matches_dict, expect_hid_equals=False)
