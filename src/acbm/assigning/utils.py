from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd


def _map_time_to_day_part(
    minutes: int, time_of_day_mapping: Optional[dict] = None
) -> str:
    """
    Map time in minutes to part of the day. The mapping dictionary is defined internally

    Parameters
    ----------
    minutes: int
        The time in minutes since midnight
    time_of_day_mapping: dict
        A dictionary mapping time range in minutes to part of the day. If not provided, the default dictionary is used

    Returns
    -------
    str
        The part of the day the time falls into
    """

    # default dictionary mapping time range in minutes to part of the day
    default_time_of_day_mapping = {
        range(300): "night",  # 00:00 - 04:59
        range(300, 720): "morning",  # 05:00 - 11:59
        range(720, 1080): "afternoon",  # 12:00 - 17:59
        range(1080, 1440): "evening",  # 18:00 - 23:59
    }

    # use the default dictionary if none is provided
    if time_of_day_mapping is None:
        time_of_day_mapping = default_time_of_day_mapping

    for time_range, description in time_of_day_mapping.items():
        if minutes in time_range:
            return description
    return "unknown"  # return 'unknown' if the time doesn't fall into any of the defined ranges


def _map_day_to_wkday_binary(day: int) -> int:
    """
    Map day of the week to binary representation. 1 for weekday, 0 for weekend

    Parameters
    ----------
    day: str
        The day of the week (1-7)

    Returns
    -------
    int
        1 if the day is a weekday, 0 if the day is a weekend
    """
    # if day in range 1: 5, it is a weekday
    if day in [1, 2, 3, 4, 5]:
        return 1
    if day in [6, 7]:
        return 0
    # if day is not in the range 1: 7, raise an error
    error_message = "Day should be numeric and in the range 1-7"
    raise ValueError(error_message)


# function to filter travel_times df specifically for pt mode
# match time_of_day to pt option


def get_travel_times_pt(
    activity: pd.Series, travel_times: pd.DataFrame
) -> pd.DataFrame:
    """

    Return the travel time results that match the time of day and whether it's a weekday / weekend
    This function is specific to public transport as bus/rail timetables vary, whereas car, walk,
    cycle trips do not (#TODO: until we add congestion)

    Parameters
    ----------

    activity: pd.Series
        the row from activity_chains that we are analysing. It should have: 'tst' (start time) and 'TravDay' (day of week)
    travel_times: pd.DataFrame
        the travel_times df. It has times between all OD pairs by mode, time_of_day and weekday/weekend

    Returns
    -------
    pd.Dataframe
        All the rows from travel_times that correspond to the time of day and weekday/weekend

    """

    # use map_time_to_day_part to add a column to activity
    activity["time_of_day"] = _map_time_to_day_part(activity["tst"])
    # use map_day_to_wkday_binary to identify if activity is on a weekday or weekend
    activity["weekday"] = _map_day_to_wkday_binary(activity["TravDay"])

    # if weekday = 1, filter travel_times df to rows where combination contains pt_wkday and the time_of_day
    if activity["weekday"] == 1:
        travel_times_filtered = travel_times[
            (travel_times["combination"].str.contains("pt_wkday"))
            & (travel_times["combination"].str.contains(activity["time_of_day"]))
        ]
    # if weekday = 0, filter travel_times df to rows where combination contains pt_wkend
    # TODO: get travel_times for all times during the weekend (morning, afternoon, evening, night)
    # and apply the same logic as weekday
    else:
        travel_times_filtered = travel_times[
            (travel_times["combination"].str.contains("pt_wkend"))
        ]

    return travel_times_filtered

    # TODO: if travel_times_filtered is empty, select the zones that are within the reported distance "TripDisIncSW"


def get_activities_per_zone(
    zones: gpd.GeoDataFrame,
    zone_id_col: str,
    activity_pts: gpd.GeoDataFrame,
    return_df: bool = False,
) -> dict:
    """
    This funciton returns the total: (a) no. of activities and (b) floorspace for each unique activity type in the activity_pts layer

    Parameters
    ----------
    zones: GeoDataFrame
        The zones GeoDataFrame
    zone_id_col: str
        The column name that contains the unique identifier for each zone.
        This column is used to group the data, and should be consistent with zone
        columns used in other parts of the pipeline
    activity_pts: GeoDataFrame
        A point layer with the location of the activities. If produced from osmox, it should have floorspace as well
    return_df: bool
        If True, the function will return a long dataframe with the columns: zone_id_col | counts | floor_area | activity
        If False, the function will return a dictionary where each element is a dataframe for an activity type with the
        following columns:
        zone_id_col | {activity_type}_counts | {activity_type}_floor_area

    Returns
    -------
    dict
        A dictionary where each element is a dataframe for an activity type with the following columns:

        zone_id_col | {activity_type}_counts | {activity_type}_floor_area
    """

    # check the crs of the two spatial layers.

    if zones.crs != activity_pts.crs:
        error_message = "The CRS of 'zones' and 'activity_pts' must be the same."
        raise ValueError(error_message)

    # create a spatial join to identify which zone each point from activity_pts is in
    activity_pts_joined = gpd.sjoin(
        activity_pts, zones[[zone_id_col, "geometry"]], how="inner", predicate="within"
    )

    # get a list of all unique activities in activity_pts (I need this step because an entry could be ['work', 'shop'])
    activity_pts_copy = activity_pts_joined["activities"].apply(
        lambda x: [i.strip() for i in x.split(",")]
    )
    activity_types = activity_pts_copy.explode().unique()

    # Iterate over each activity type, and create a new boolean column "has_{activity}" to indicate the presence of an activity
    for activity in activity_types:
        activity_pts_joined[f"has_{activity}"] = activity_pts_joined[
            "activities"
        ].apply(lambda x, activity=activity: activity in x)

    # group the data by "has_{activity}" and the zone id, and get the number and floorspace of each activity type in each zone
    # the output is a dictionary of dfs (one for each activity_type)
    grouped_data = {
        activity: (
            activity_pts_joined[activity_pts_joined[f"has_{activity}"]]
            .groupby(zone_id_col)
            .agg({f"has_{activity}": "sum", "floor_area": "sum"})
            .reset_index()
            .rename(
                columns={
                    f"has_{activity}": f"{activity}_counts",
                    "floor_area": f"{activity}_floor_area",
                }
            )
        )
        for activity in activity_types
    }

    if return_df:
        return _get_activities_per_zone_df(grouped_data)
    # if return_df is False, return the dictionary
    return grouped_data


def _get_activities_per_zone_df(activities_per_zone: dict) -> pd.DataFrame:
    """
    This is an internal function to use inside the get_activities_per_zone function.
    get_activities_per_zone returns a dictionary of dataframes. This function concatenates
    the dataframes into a single dataframe for easier processing

    Parameters
    ----------
    activities_per_zone: dict
        A dictionary where each element is a dataframe for an activity type with the following columns:
        zone_id_col | {activity_type}_counts | {activity_type}_floor_area

    Returns
    -------
    pd.DataFrame
        A long dataframe with the columns: zone_id_col | counts | floor_area | activity

    """
    # Create a long df with all the data (for filtering)

    # For each df in activities per zone, rename the columns ending with "_counts" to counts
    # and the columns ending with "_floor_area" to floor_area
    for activity, df in activities_per_zone.items():
        new_columns = []
        for col in df.columns:
            if "_counts" in col:
                new_columns.append("counts")
            elif "_floor_area" in col:
                new_columns.append("floor_area")
            else:
                new_columns.append(col)
        df.columns = new_columns
        # add a column for the activity type
        df["activity"] = activity

    # concatenate all the dataframes in activities_per_zone
    return pd.concat(activities_per_zone.values())


def zones_to_time_matrix(
    zones: gpd.GeoDataFrame,
    id_col: Optional[str] = None,
    to_dict: Optional[bool] = False,
) -> dict:
    """
    Calculates the distance matrix between the centroids of the given zones and returns it as a DataFrame. The matrix also adds
    a column for all the different modes, with travel time (seconds) based on approximate speeds

    The function first converts the CRS of the zones to EPSG:27700 and calculates the centroids of the zones.
    Then, it calculates the distance matrix between the centroids and reshapes it from a wide format to a long format.
    If an id_col is specified, the function replaces the index values in the distance matrix with the corresponding id_col values from the zones.

    Parameters
    ----------
    zones: (gpd.GeoDataFrame):
        A GeoDataFrame containing the zones.
    id_col (str, optional):
        The name of the column in the zones GeoDataFrame to use as the ID. If None, the index values are used. Default is None.

    Returns
    -------
    if to_dict = False
        pd.DataFrame: A dataframe containing the distance matrix. The columns are:
    '{id_col}_from', '{id_col}_to', 'distance', {time}_{mode} with a column for each mode in the dictionary
    if to_dict = True
        converts the data to a dictionary
        keys: a tuple representing ({id_col}_from, {id_col}_to)
        values: another dictionary with the following keys:
            - 'distance': a float representing the distance between the two locations.
            - 'time_car': a float representing the travel time by car between the two locations.
            - 'time_pt': a float representing the travel time by public transport between the two locations.
            - 'time_cycle': a float representing the travel time by bicycle between the two locations.
            - 'time_walk': a float representing the travel time on foot between the two locations.
            - 'time_average': a float representing the average travel time between the two locations.
        a value can be accessed using eg: dict[({id_col}_from, {id_col}_to)]['time_car']
    """

    zones = zones.to_crs(epsg=27700)
    centroids = zones.geometry.centroid

    # get distance matrix (meters)
    distances = centroids.apply(lambda g: centroids.distance(g))
    # wide to long
    distances = distances.stack().reset_index()
    distances.columns = [f"{id_col}_from", f"{id_col}_to", "distance"]

    # replace the index values with the id_col values if id_col is specified
    if id_col is not None:
        # create a mapping from index to id_col values
        id_mapping = zones[id_col].to_dict()

        # replace
        distances[f"{id_col}_from"] = distances[f"{id_col}_from"].map(id_mapping)
        distances[f"{id_col}_to"] = distances[f"{id_col}_to"].map(id_mapping)

    # define speed by mode
    mode_speeds_mps = {
        "car": 20 * 1000 / 3600,
        "pt": 15 * 1000 / 3600,
        "cycle": 15 * 1000 / 3600,
        "walk": 5 * 1000 / 3600,
        "average": 15 * 1000 / 3600,
    }

    # add travel time estimates (per mode)
    for mode, speed in mode_speeds_mps.items():
        distances[f"time_{mode}"] = distances["distance"] / speed

    if to_dict:
        # convert to a dictionary
        distances_dict = distances.set_index(
            [f"{id_col}_from", f"{id_col}_to"]
        ).to_dict("index")

    return distances_dict


def filter_matrix_to_boundary(
    boundary,
    matrix,
    boundary_id_col,
    matrix_id_col,
    matrix_id_col_sfx: Optional[List[str]] = None,
    type="both",
) -> pd.DataFrame:
    """
    Filter the matrix to only include rows and columns that are in the boundary. We can filter
    based on matching origin, destination, or both.

    Parameters
    ----------
    boundary : GeoDataFrame
        The boundary GeoDataFrame.
    matrix : DataFrame
        The matrix DataFrame.
    boundary_id_col : str
        The column name in the boundary GeoDataFrame that contains the unique identifier.
    matrix_id_col : str
        The column name in the matrix DataFrame that contains the unique identifier.
    matrix_id_cols_sfx : list, optional
        The suffixes to add to the matrix_id_col to get the origin and destination columns.
        The default is ["_home", "_work"].
    type : str, optional
        The type of filtering to apply. Options are 'origin', 'destination', 'columns'. The default is 'both'

    Returns
    -------

    filtered_matrix : DataFrame
        The filtered matrix DataFrame.
    """

    if matrix_id_col_sfx is None:
        matrix_id_col_sfx = ["_home", "_work"]

    matrix_id_col_from = matrix_id_col + matrix_id_col_sfx[0]
    matrix_id_col_to = matrix_id_col + matrix_id_col_sfx[1]

    if type == "origin":
        filtered_matrix = matrix[
            matrix[matrix_id_col_from].isin(boundary[boundary_id_col])
        ]

    elif type == "destination":
        filtered_matrix = matrix[
            matrix[matrix_id_col_to].isin(boundary[boundary_id_col])
        ]

    elif type == "both":
        filtered_matrix = matrix[
            matrix[matrix_id_col_from].isin(boundary[boundary_id_col])
            & matrix[matrix_id_col_to].isin(boundary[boundary_id_col])
        ]

    return filtered_matrix


def intrazone_time(zones: gpd.GeoDataFrame) -> dict:
    """
    Estimate the time taken to travel within each zone.

    The function calculates the area of each zone, and assumes that they are regular polygons. We then
    assume that the average travel distance within the zone is equal to the 'radius' of the zone.
    Travel time is based on estimated speed by mode

    Parameters
    ----------
    zones : gpd.GeoDataFrame
        The GeoDataFrame containing the zones with zone IDs as the GeoDataFrame index.

    Returns
    -------
    dict
        A dictionary containing the intrazone travel time estimates.
        Example row:
        {53506: {'car': 0.3, 'pt': 0.5, 'cycle': 0.5, 'walk': 1.4, 'average': 0.5},
    """

    # convert zones to metric crs
    zones = zones.to_crs(epsg=27700)
    # get the sqrt of the area of each zone
    zones["area"] = zones["geometry"].area
    zones["average_dist"] = np.sqrt(zones["area"]) / 2

    # mode speeds in m/s
    mode_speeds_mps = {
        "car": 20 * 1000 / 3600,
        "pt": 15 * 1000 / 3600,
        "cycle": 15 * 1000 / 3600,
        "walk": 5 * 1000 / 3600,
        "average": 15 * 1000 / 3600,
    }

    # Create (and return) a dictionary where key = zone and values = travel time in minutes
    return {
        zone: {
            mode: round((dist / speed) / 60, 1)
            for mode, speed in mode_speeds_mps.items()
        }
        for zone, dist in zones["average_dist"].items()
    }


def replace_intrazonal_travel_time(
    travel_times: pd.DataFrame, intrazonal_estimates: dict, column_to_replace: str
) -> pd.DataFrame:
    """
    Replace the intrazonal travel times in a travel time matrix.

    Intrazonal travel times from routing engines (e.g. r5) are normally 0. We replace these with estimates
    based on the area of the zone ( values calculated using intrazone_time() function).

    Parameters
    ----------
    travel_times : pd.DataFrame
        The DataFrame containing the travel time estimates. It will be modified
    intrazonal_estimates : dict
        The dictionary containing the intrazonal travel time estimates. From intrazone_time() function.
    column_to_replace : str
        The name of the column wth the travel time estimates to replace.

    Returns
    -------

    pd.DataFrame
        A DataFrame with the intrazonal travel time estimates replaced.
    """

    # Copy the DataFrame to avoid modifying the original one
    travel_times_copy = travel_times.copy()

    # Create a new column 'mode' by splitting the 'combination' column
    travel_times_copy["mode"] = travel_times_copy["combination"].str.split("_").str[0]

    # Create a mask for intrazonal trips
    intrazonal_mask = travel_times_copy["from_id"] == travel_times_copy["to_id"]

    # Apply the intrazonal estimates using a vectorized operation
    travel_times_copy.loc[intrazonal_mask, column_to_replace] = travel_times_copy[
        intrazonal_mask
    ].apply(lambda row: intrazonal_estimates[row["from_id"]][row["mode"]], axis=1)

    # Return the modified DataFrame
    return travel_times_copy
