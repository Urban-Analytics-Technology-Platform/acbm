import pandas as pd
import polars as pl
import pytest

from acbm.assigning.utils import get_chosen_day
from acbm.config import load_config
from acbm.utils import (
    households_with_common_travel_days,
    households_with_travel_days_in_nts_weeks,
)


@pytest.fixture
def config():
    return load_config("config/base.toml")


@pytest.fixture
def nts_trips():
    return pd.DataFrame(
        [
            [1, 1, 1],
            [2, 1, 1],
            [3, 1, 1],
            [4, 2, 2],
            [4, 2, 4],
            [5, 2, 3],
            [6, 3, 3],
            [7, 3, 3],
            [8, 3, 3],
            [9, 3, 3],
            [10, 4, pd.NA],
            [11, 4, pd.NA],
            [12, 4, pd.NA],
            [13, 5, pd.NA],
            [14, 5, pd.NA],
            [15, 5, 4],
            [16, 5, 4],
            [17, 6, 5],
            [18, 6, 5],
            [19, 6, pd.NA],
            [19, 6, 4],
        ],
        columns=["IndividualID", "HouseholdID", "TravDay"],
    )


@pytest.fixture
def nts_trips_with_aliases(nts_trips):
    df = nts_trips
    df["id"] = df["IndividualID"]
    df["household"] = df["HouseholdID"]
    return df


def test_households_with_common_travel_days(nts_trips):
    assert households_with_common_travel_days(nts_trips, [1]) == [1]
    assert households_with_common_travel_days(nts_trips, [1, 2]) == [1]
    assert households_with_common_travel_days(nts_trips, [1, 3]) == [1, 3]
    assert households_with_common_travel_days(nts_trips, [1, 3, 4]) == [1, 3]


def test_households_with_travel_days_in_nts_weeks(nts_trips):
    assert households_with_travel_days_in_nts_weeks(nts_trips, [1]) == [1]
    assert households_with_travel_days_in_nts_weeks(nts_trips, [1, 2]) == [1]
    assert households_with_travel_days_in_nts_weeks(nts_trips, [1, 3]) == [1, 3]
    assert households_with_travel_days_in_nts_weeks(nts_trips, [1, 3, 4]) == [1, 2, 3]
    assert households_with_travel_days_in_nts_weeks(nts_trips, [1, 3, 4, 5]) == [
        1,
        2,
        3,
        6,
    ]


def test_get_chosen_day_with_common_travel_day(nts_trips_with_aliases):
    pl.set_random_seed(0)
    hids = households_with_common_travel_days(nts_trips_with_aliases, [1, 3, 4])
    nts_trips_with_aliases = nts_trips_with_aliases[
        nts_trips_with_aliases["household"].isin(hids)
    ]
    df = get_chosen_day(nts_trips_with_aliases, True)
    print(df)
    assert df.to_numpy().prod(1).sum() == 96


def test_get_chosen_day_with_travel_days_in_nts_weeks(nts_trips_with_aliases):
    pl.set_random_seed(0)
    hids = households_with_travel_days_in_nts_weeks(
        nts_trips_with_aliases, [1, 3, 4, 5]
    )
    nts_trips_with_aliases = nts_trips_with_aliases[
        nts_trips_with_aliases["household"].isin(hids)
    ]
    df = get_chosen_day(nts_trips_with_aliases, False)
    print(df)
    assert df.to_numpy().prod(1).sum() == 370
