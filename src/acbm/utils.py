import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error

from acbm.config import Config


def calculate_rmse(predictions, targets):
    """
    Calculate the Root Mean Squared Error (RMSE) between predictions and targets,
    excluding NaN values.

    Parameters:
    - predictions (np.ndarray or pd.Series): The predicted values.
    - targets (np.ndarray or pd.Series): The actual values.

    Returns:
    - float: The RMSE value.
    """
    # Ensure inputs are numpy arrays
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Create a mask for non-NaN values
    mask = ~np.isnan(predictions) & ~np.isnan(targets)

    # Filter out NaN values
    filtered_predictions = predictions[mask]
    filtered_targets = targets[mask]

    # Calculate mean squared error
    mse = mean_squared_error(filtered_predictions, filtered_targets)

    # Calculate and return RMSE
    return np.sqrt(mse)


def get_travel_times(config: Config, use_estimates: bool = False) -> pd.DataFrame:
    if config.parameters.travel_times and not use_estimates:
        return pd.read_parquet(config.travel_times_filepath)
    return pd.read_parquet(config.travel_times_estimates_filepath)


def households_with_common_travel_days(
    nts_trips: pd.DataFrame, days: list[int], hid="HouseholdID", pid="IndividualID"
) -> list[int]:
    return (
        pl.DataFrame(nts_trips)
        .lazy()
        # group_by household and individual
        .group_by([hid, pid])
        # get unique travel days
        .agg(pl.col("TravDay").unique())
        # group by household
        .group_by(hid)
        # create aggregates for:
        # - unique days
        # - counts of each unique day across household
        # - number of individuals in household
        .agg(
            [
                pl.col("TravDay").explode().unique(),
                pl.col("TravDay").explode().unique_counts().alias("day_count"),
                pl.col(hid).count().alias("count"),
            ]
        )
        # explode lists
        .explode(["TravDay", "day_count"])
        # filter for days that are common across all individuals
        .filter(pl.col("day_count").eq(pl.col("count")))
        # filter for days in given set of days
        .filter(pl.col("TravDay").is_in(days))
        .collect()
        # return list of unique household ids
        .get_column(hid)
        .unique()
        .sort()
        .to_list()
    )


def households_with_travel_days_in_nts_weeks(
    nts_trips: pd.DataFrame, days: list[int], hid="HouseholdID", pid="IndividualID"
) -> list[int]:
    return (
        pl.DataFrame(nts_trips)
        .group_by([hid, pid])
        .agg(pl.col("TravDay").unique())
        .select(
            [
                hid,
                pid,
                pl.col("TravDay").list.drop_nulls().list.set_intersection(pl.lit(days)),
            ]
        )
        .select(
            [
                hid,
                pid,
                pl.when(pl.col("TravDay").list.len().eq(0))
                .then(None)
                .otherwise(pl.col("TravDay"))
                .alias("TravDay"),
            ]
        )
        .group_by(hid)
        .agg(pl.col("TravDay").list.len().ne(0).all())
        .filter(pl.col("TravDay").eq(True))
        .get_column(hid)
        .sort()
        .to_list()
    )
