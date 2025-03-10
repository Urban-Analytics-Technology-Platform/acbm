import numpy as np
import pandas as pd
import pytest

from acbm.assigning.utils import _adjust_distance, _map_day_to_wkday_binary

# applying to a single value


@pytest.mark.parametrize(
    ("day", "expected"), [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 0), (7, 0)]
)
def test_map_day_to_wkday_binary_valid(day, expected):
    assert _map_day_to_wkday_binary(day) == expected


def test_map_day_to_wkday_binary_invalid():
    with pytest.raises(ValueError, match="Day should be numeric and in the range 1-7"):
        _map_day_to_wkday_binary(8)


# applying to a df


def test_map_day_to_wkday_binary_df():
    df = pd.DataFrame({"day": [1, 2, 3, 4, 5, 6, 7]})
    df["wkday"] = df["day"].apply(_map_day_to_wkday_binary)
    assert df["wkday"].tolist() == [1, 1, 1, 1, 1, 0, 0]


# test applying the function to a df column with an invalid value
def test_map_day_to_wkday_binary_df_invalid():
    df = pd.DataFrame({"day": [1, 2, 3, 4, 5, 6, 7, 8]})
    with pytest.raises(ValueError, match="Day should be numeric and in the range 1-7"):
        df["wkday"] = df["day"].apply(_map_day_to_wkday_binary)


def test_adjust_distance():
    assert np.isclose(_adjust_distance(0, 1.5, 0.1), 0)
    assert np.isclose(_adjust_distance(10000, 1.56, 0.0001), 12060.125)
    assert np.isclose(
        _adjust_distance(10000, 1.56, 0.0001),
        10000 * (1 + ((1.56 - 1) * np.exp(-0.0001 * 10000))),
    )
    assert np.isclose(_adjust_distance(50000, 1.8, 0.0002), 50001.81599)
    assert np.isclose(
        _adjust_distance(50000, 1.8, 0.0002),
        50000 * (1 + ((1.8 - 1) * np.exp(-0.0002 * 50000))),
    )
