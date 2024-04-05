import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# categorical (exact) matching


def match_categorical(
    df_pop: pd.DataFrame,
    df_pop_cols: list,
    df_pop_id: str,
    df_sample: pd.DataFrame,
    df_sample_cols: list,
    df_sample_id: str,
    chunk_size: int,
    show_progress=True,
) -> dict:
    """
    Match the rows in two DataFrames based on specified columns.
    The function matches the rows in df_pop to the rows in df_sample based
    on the columns in df_pop_cols and df_sample_cols. The matching is done
    in chunks to avoid memory issues.

    Parameters
    ----------
    df_pop: pandas DataFrame
        The DataFrame to be matched on
    df_pop_cols: list
        The columns to be used for matching in df_pop
    df_pop_id: str
        The column name that contains the unique identifier in df_pop
        It is the key in the final dictionary
    df_sample: pandas DataFrame
        The DataFrame to be matched with
    df_sample_cols: list
        The columns to be used for matching in df_sample
    df_sample_id: str
        The column name that contains the unique identifier in df_sample
        It is the value in the final dictionary
    chunk_size: int
        The number of rows to process at a time
    show_progress: bool
        Whether to print the progress of the matching to the console

    Returns
    -------
    results: dict
        A dictionary with the matched rows {df_pop_id: [df_sample_id]}

    """

    # dictionary to store results
    results = {}

    # loop over the df_pop DataFrame in chunks
    for i in range(0, df_pop.shape[0], chunk_size):
        # filter the df_pop DataFrame to the current chunk
        j = i + chunk_size
        if show_progress:
            print("matching rows ", i, "to", j, " out of ", df_pop.shape[0])

        df_pop_chunk = df_pop.iloc[i:j]

        # merge the df_pop_chunk with the df_sample DataFrame
        df_matched_chunk = df_pop_chunk.merge(
            df_sample, left_on=df_pop_cols, right_on=df_sample_cols, how="left"
        )

        # convert the matched df to a dictionary:
        df_matched_dict_i = (
            df_matched_chunk.groupby(df_pop_id)[df_sample_id].apply(list).to_dict()
        )

        # add the dictionary to results{}
        results.update(df_matched_dict_i)
    return results


# propensity score matching


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

    # Initialize an empty dict to store the matches
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

        # Get the row id from df1 and df2
        row_id_df1 = df1.index[min_distance_index]
        row_id_df2 = df2.index[closest_df2_index]

        # Store the match in the dictionary
        matches[row_id_df1] = row_id_df2

        # Remove the matched rows from df1 and df2
        df1 = df1.drop(df1.index[min_distance_index])
        df2 = df2.drop(df2.index[closest_df2_index])

    return matches


# TODO: parallelize the matching process. See this stackoverflow suggestion
# for iterating over dict keys https://stackoverflow.com/a/30075659
def match_individuals(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    matching_columns: list,
    df1_id: str,
    df2_id: str,
    matches_hh: dict,
    show_progress: bool = False,
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
    show_progress: bool
        Whether to print the progress of the matching to the console

    Returns
    -------
    matches: dict
        A dictionary with the matched row indeces from the two DataFrames {df1: df2}

    """
    # Initialize an empty dic to store the matches
    matches = {}
    # Remove all unmateched households
    matches_hh = {key: value for key, value in matches_hh.items() if not pd.isna(value)}

    # loop over all rows in the matches_hh dictionary
    for i, (key, value) in enumerate(matches_hh.items(), 1):
        # Get the rows in df1 and df2 that correspond to the matched hids
        rows_df1 = df1[df1[df1_id] == key]
        rows_df2 = df2[df2[df2_id] == int(value)]

        if show_progress:
            # Print the iteration number and the number of keys in the dict
            print(f"Matching for household {i} out of: {len(matches_hh)}")

        # apply the matching
        match = match_psm(rows_df1, rows_df2, matching_columns)

        # append the results to the main dict
        matches.update(match)

    return matches
