import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def match_psm(df1: pd.DataFrame, df2: pd.DataFrame, matching_columns: list) -> dict:
    """
    Use the Propensity Score Matching (PSM) method to match the rows in two DataFrames
    The distances between columns is calculated using the NearestNeighbors algorithm

    Parameters
    ----------
    df1: pandas DataFrame
        The first DataFrame to be matched on
    df2: pandas DataFrame
        The second DataFrame to be matched with
    matching_columns: list
        The columns to be used for the matching

    Returns
    -------
    matches: dict
        A dictionary with the matched row indeces from the two DataFrames {df1: df2}
    """

    # Initialize an empty dic to store the matches
    matches = {}

    # Matching without replacement
    while not df1.empty:
        # Fit a NearestNeighbors model on the specified columns for df2
        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        nn.fit(df2[matching_columns])

        # Find the closest row in df2 for each row in df1
        distances, indices = nn.kneighbors(df1[matching_columns])

        # Get the index of the closest match in df2 for each row in df1
        closest_indices = indices.flatten()

        # Get the row in df1 with the smallest distance to its closest match in df2
        min_distance_index = np.argmin(distances)

        # Get the corresponding row in df2
        closest_df2_index = closest_indices[min_distance_index]

        # Get the hid from df1 and df2
        hid_df1 = df1.index[min_distance_index]
        hid_df2 = df2.index[closest_df2_index]

        # Store the match in the dictionary
        matches[hid_df1] = hid_df2

        # Remove the matched rows from df1 and df2
        df1 = df1.drop(df1.index[min_distance_index])
        df2 = df2.drop(df2.index[closest_df2_index])

    return matches


def match_individuals(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    matching_columns: list,
    df1_id: str,
    df2_id: str,
    matches_hh: dict,
) -> dict:
    """
    Apply a matching function iteratively to members of each household. 
    In each iteration, filter df1 and df2 to the household ids of item i 
    in matches_hh, and then apply the matching function to the filtered DataFrames.

    Parameters
    ----------
    df1: pandas DataFrame
        The first DataFrame to be matched on
    df2: pandas DataFrame
        The second DataFrame to be matched with
    matching_columns: list
        The columns to be used for the matching
    df1_id: str
        The household_id from the first DataFrame
    df2_id: str
        The household_id from the second DataFrame
    matches_hh: dict
        A dictionary with the matched household ids {df1_id: df2_id}

    Returns
    -------
    matches: dict
        A dictionary with the matched row indeces from the two DataFrames {df1: df2}

    """
    # Initialize an empty dic to store the matches
    matches = {}

    # loop over all rows in the matches_hh dictionary
    for i, (key, value) in enumerate(matches_hh.items(), 1):
        # Get the rows in df1 and df2 that correspond to the matched hids
        rows_df1 = df1[df1[df1_id] == key]
        rows_df2 = df2[df2[df2_id] == int(value)]

        # Print the iteration number and the number of keys in the dict
        print(f"Matching for household {i} out of: {len(matches_hh)}")

        # apply the matching
        match = match_psm(rows_df1, rows_df2, matching_columns)

        # append the results to the main dict
        matches.update(match)

    return matches
