import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from pandarallel import pandarallel

# Define logger at the module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a handler that outputs to the console
console_handler = logging.StreamHandler()
# Create a handler that outputs to a file
file_handler = logging.FileHandler("log_assigning.log")


# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

pandarallel.initialize(progress_bar=True)


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
    raise ValueError("Day should be numeric and in the range 1-7")


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


def get_possible_zones(
    activity_chains: pd.DataFrame,
    travel_times: pd.DataFrame,
    activities_per_zone: pd.DataFrame,
    activity_col: str,
    filter_by_activity: bool = False,
    time_tolerance: int = 0.2,
) -> dict:
    """
    Get possible zones for all activity chains in the dataset. This function loops over the travel_times dataframe and filters by mode, time of day and weekday/weekend.
    At each loop it applies the _get_possible_zones function to each row in the activity_chains dataframe.
    The travel_times dataframe is big, so doing some initial filtering before applying _get_possible_zones makes the process faster as less filtering is done for each
    row when running _get_possible_zones.

    Parameters
    ----------
    activity_chains: pd.DataFrame
        A dataframe with activity chains
    travel_times: pd.DataFrame
        A dataframe with travel times between zones
    activities_per_zone: pd.DataFrame
        A dataframe with the number of activities and floorspace for each zone. The columns are 'OA21CD', 'counts', 'floor_area', 'activity'
        where 'activity' is the activity type as defined in the osmox config file
    filter_by_activity: bool
        If True, we will return a results that only includes destination zones that have an activity that matches the activity purpose
    time_tolerance: int
        The time tolerance is used to filter the travel_times dataframe to only include travel times within a certain range of the
        activity chain's travel time (which is stored in "TripTotalTime"). Allowable travel_times are those that fall in the range of:
        travel_time_reported * (1 - time_tolerance) <= travel_time_reported <= travel_time_reported * (1 + time_tolerance)
        Default = 0.2

    Returns
    -------
    dict
        A dictionary of dictionaries. Each dictionary is for one of the rows in activity chains
        with the origin zone as the key and a list of possible destination zones as the value
    """
    list_of_modes = activity_chains["mode"].unique()
    print(f"Unique modes found in the dataset are: {list_of_modes}")

    # use map_day_to_wkday_binary to identify if activity is on a weekday or weekend
    activity_chains["weekday"] = activity_chains["TravDay"].apply(
        _map_day_to_wkday_binary
    )
    # day types identifier (weekday/weekend)
    day_types = activity_chains["weekday"].unique()
    # use map_time_to_day_part to add a column to activity
    activity_chains["time_of_day"] = activity_chains["tst"].apply(_map_time_to_day_part)
    # get unique time_of_day values
    list_of_times_of_day = activity_chains["time_of_day"].unique()

    # create an empty dictionary to store the results
    results = {}

    # loop over the list of modes
    for mode in list_of_modes:
        print(f"Processing mode: {mode}")
        # filter the travel_times dataframe to only include rows with the current mode
        travel_times_filtered_mode = travel_times[
            travel_times["combination"].apply(lambda x: x.split("_")[0]) == mode
        ]

        # if the mode is public transport, we need to filter the travel_times data based on time_of_day and weekday/weekend
        if mode == "pt":
            for time_of_day in list_of_times_of_day:
                print(f"Processing time of day: {time_of_day} | mode: {mode}")
                for day_type in day_types:
                    if day_type == 1:
                        print(
                            f"Processing time of day: {time_of_day} | weekday: {day_type} | mode: {mode}"
                        )
                        # filter the travel_times dataframe to only include rows with the current time_of_day and weekday
                        travel_times_filtered_mode_time_day = (
                            travel_times_filtered_mode[
                                (
                                    travel_times_filtered_mode[
                                        "combination"
                                    ].str.contains("pt_wkday")
                                )
                                & (
                                    travel_times_filtered_mode[
                                        "combination"
                                    ].str.contains(time_of_day)
                                )
                            ]
                        )
                        print(
                            "unique modes after filtering are",
                            travel_times_filtered_mode_time_day["combination"].unique(),
                        )
                    elif day_type == 0:
                        print(
                            f"Processing time of day: {time_of_day} | weekday: {day_type} | mode: {mode}"
                        )
                        travel_times_filtered_mode_time_day = (
                            travel_times_filtered_mode[
                                (
                                    travel_times_filtered_mode[
                                        "combination"
                                    ].str.contains("pt_wkend")
                                )
                                & (
                                    travel_times_filtered_mode[
                                        "combination"
                                    ].str.contains(time_of_day)
                                )
                            ]
                        )
                        print(
                            "unique modes after filtering are",
                            travel_times_filtered_mode_time_day["combination"].unique(),
                        )

                    # filter the activity chains to the current mode, time_of_day and weekday
                    activity_chains_filtered = activity_chains[
                        (activity_chains["mode"] == mode)
                        & (activity_chains["time_of_day"] == time_of_day)
                        & (activity_chains["weekday"] == day_type)
                    ]

                    if (
                        not travel_times_filtered_mode_time_day.empty
                        and not activity_chains_filtered.empty
                    ):
                        # apply get_possible_zones to each row in activity_chains_filtered
                        # pandarallel.initialize(progress_bar=True)
                        possible_zones = activity_chains_filtered.parallel_apply(
                            lambda row,
                            tt=travel_times_filtered_mode_time_day: _get_possible_zones(
                                activity=row,
                                travel_times=tt,
                                activities_per_zone=activities_per_zone,
                                filter_by_activity=filter_by_activity,
                                activity_col=activity_col,
                                time_tolerance=time_tolerance,
                            ),
                            axis=1,
                        )

                        results.update(possible_zones)

        # for all other modes, we don't care about time of day and weekday/weekend
        else:
            travel_times_filtered_mode_time_day = travel_times_filtered_mode
            activity_chains_filtered = activity_chains[
                (activity_chains["mode"] == mode)
            ]

            if (
                not travel_times_filtered_mode_time_day.empty
                and not activity_chains_filtered.empty
            ):
                # apply _get_possible_zones to each row in activity_chains_filtered
                # pandarallel.initialize(progress_bar=True)
                possible_zones = activity_chains_filtered.parallel_apply(
                    lambda row,
                    tt=travel_times_filtered_mode_time_day: _get_possible_zones(
                        activity=row,
                        travel_times=tt,
                        activities_per_zone=activities_per_zone,
                        filter_by_activity=filter_by_activity,
                        activity_col=activity_col,
                        time_tolerance=time_tolerance,
                    ),
                    axis=1,
                )

                results.update(possible_zones)

    return results


def _get_possible_zones(
    activity: pd.Series,
    travel_times: pd.DataFrame,
    activities_per_zone: pd.DataFrame,
    filter_by_activity: bool,
    activity_col: str,
    time_tolerance: int = 0.2,
) -> dict:
    """
    Get possible zones for a given activity chain

    Parameters
    ----------
    activity: pd.Series
        A row from the activity chains dataframe. It should contain the following columns: 'tst', 'TripTotalTime', 'mode', 'OA21CD'
    travel_times: pd.DataFrame
        A dataframe with travel times between zones
    activities_per_zone: pd.DataFrame
        A dataframe with the number of activities and floorspace for each zone. The columns are 'OA21CD', 'counts', 'floor_area', 'activity'
        where 'activity' is the activity type as defined in the osmox config file
    filter_by_activity: bool
        If True, we will return a results that only includes destination zones that have an activity that matches the activity purpose
    time_tolerance: int
        The time tolerance is used to filter the travel_times dataframe to only include travel times within a certain range of the
        activity chain's travel time (which is stored in "TripTotalTime"). Allowable travel_times are those that fall in the range of:
        travel_time_reported * (1 - time_tolerance) <= travel_time_reported <= travel_time_reported * (1 + time_tolerance)
        Default = 0.2

    Returns
    -------
    dict
        A dictionary with the origin zone as the key and a list of possible destination zones as the value
    """

    # get the travel time
    travel_time = activity["TripTotalTime"]
    # get the origin zone
    origin_zone = activity["OA21CD"]
    # get the activity purpose
    activity_purpose = activity[activity_col]

    # filter the travel_times dataframe by trip_origin and activity_purpose
    travel_times_filtered_origin_mode = travel_times[
        travel_times["OA21CD_from"] == origin_zone
    ]
    # do we include only zones that have an activity that matches the activity purpose?
    if filter_by_activity:
        filtered_activities_per_zone = activities_per_zone[
            # activities_per_zone["activity"].str.split("_").str[0] == activity_purpose
            activities_per_zone["activity"] == activity_purpose
        ]
        logger.debug(
            f"Activity {activity.id}: Number of zones with activity {activity_purpose}: \
            {len(filtered_activities_per_zone)}"
        )

        # keep only the zones that have the activity purpose
        travel_times_filtered_origin_mode = travel_times_filtered_origin_mode[
            travel_times_filtered_origin_mode["OA21CD_to"].isin(
                filtered_activities_per_zone["OA21CD"]
            )
        ]
    # how many zones are reachable?
    logger.debug(
        f"Activity {activity.id}: Number of zones with activity {activity_purpose} \
        that are reachable using reported mode: {len(travel_times_filtered_origin_mode)}"
    )

    # filter by reported trip time
    travel_times_filtered_time = travel_times_filtered_origin_mode[
        (
            travel_times_filtered_origin_mode["travel_time_p50"]
            >= travel_time - time_tolerance * travel_time
        )
        & (
            travel_times_filtered_origin_mode["travel_time_p50"]
            <= travel_time + time_tolerance * travel_time
        )
    ]
    logger.debug(
        f"Activity {activity.id}: Number of zones with activity {activity_purpose} within threshold of reported time {travel_time}: \
            {len(travel_times_filtered_time)}"
    )

    # if travel_times_filtered_time returns an empty df, select the row with the closest time to the reported time
    if travel_times_filtered_time.empty:
        logger.debug(
            f"Activity {activity.id}: NO zones match activity {activity_purpose} within threshold of reported time {travel_time}: \
            Relaxing tolerance and getting matching zone that is closest to reported travel time"
        )
        travel_times_filtered_time = travel_times_filtered_origin_mode.iloc[
            (travel_times_filtered_origin_mode["travel_time_p50"] - travel_time)
            .abs()
            .argsort()[:1]
        ]

    # create dictionary with key = origin_zone and values = list of travel_times_filtered.OA21CD_to
    possible_zones = (
        travel_times_filtered_time.groupby("OA21CD_from")["OA21CD_to"]
        .apply(list)
        .to_dict()
    )

    return possible_zones


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
    activities_per_zone_df = pd.concat(activities_per_zone.values())

    return activities_per_zone_df


def select_zone(
    row: pd.Series,
    possible_zones: dict,
    activities_per_zone: pd.DataFrame,
    weighting: str = "none",
    zone_id_col: str = "OA21CD",
) -> str:
    """
    Select a zone for an activity. For each activity, we have a list of possible zones
    stored in the possible_zones dictionary. For each activity, the function will filter the
    activities_per_zone_df to the zones that (a) are in possible_zones for that activity key,
    and (b) have an activity that matches the activity in row["education_type"]. For example, if
    the activity is "education_university", the function will filter the activities_per_zone_df to
    only include zones with a university.
    The funciton relaxes this constraint if no zones match the conditions

    the next step is to sample a zone from the shortlisted zones. This is done based on the total
    floor area of available facilities in the zone (that match the activity purpose). This is from
    activities_per_zone_df["floor_area"]. Again, different options exist if floor area cannot be used

    Parameters
    ----------
    row: pd.Series
        the row that we are applying the funciton to
    possible_zones: dict
        a dictionary with keys as the index of the "row" and values as a nested dictionary
        key: Origin Zone Id, values: List of possible destination zones
    activities_per_zone: pd.DataFrame
        a dataframe with columns: OA21CD | counts | floor_area | activity
        activity is a category (e.g. work, education_school) from osm data labeling (through osmox)
        counts and floor_area are totals for a specific activity and zone
    weighting: str
        options are: "floor_area", "counts", "none"
        once we have a list of feasible zones, do we want to do weighted sampling based on
        "floor_area" or "counts" or just random sampling "none"
    zone_id: str
        The column name of the zone id in activities_per_zone. The ids should also match the nested key in
        possible_zones {key: {KEY: value}}

    Returns
    -------
    str
        The zone_id of the selected zone.
    """

    # Check if the input is in the list of allowed values
    allowed_weightings = ["floor_area", "counts", "none"]
    if weighting not in allowed_weightings:
        raise ValueError(
            f"Invalid value for weighting: {weighting}. Allowed values are {allowed_weightings}."
        )

    # get the values from possible_zones_school that match the index of the row
    # use try/except as some activities might have no possible zones
    try:
        activity_i_options = list(possible_zones[row.name].values())
        if not activity_i_options:  # Check if the list is empty
            logger.info(f"Activity {row.name}: No zones available")
            return "NA"
        # log the number of options for the specific index
        logger.debug(
            f"Activity {row.name}: Initial number of options for activity = {len(activity_i_options[0])}"
        )

        # Attempt 1: filter activities_per_zone_df to only include possible_zones
        options = activities_per_zone[
            (activities_per_zone["activity"] == row["education_type"])
            & (activities_per_zone[zone_id_col].isin(activity_i_options[0]))
        ]
        logger.debug(
            f"Activity {row.name}: Number of options after filtering by education type: {len(options)}"
        )

        # Attempt 2: if no options meet the conditions, relax the constraint by considering all education types
        # not just the education type that maps onto the person's age
        if options.empty:
            # print("No options available. Relaxing constraint")
            options = activities_per_zone[
                (activities_per_zone["activity"].str.contains("education"))
                & (activities_per_zone[zone_id_col].isin(activity_i_options[0]))
            ]
            logger.debug(
                f"Activity {row.name}: Number of options after first relaxation: {len(options)}"
            )

        # Attempt 3: if options is still empty, relax the constraint further by considering all possible zones
        # regardless of activities
        if options.empty:
            logger.info(
                f"Activity {row.name}: No zones with required facility type. Selecting from all possible zones"
            )
            options = activities_per_zone[
                activities_per_zone[zone_id_col].isin(activity_i_options[0])
            ]
            logger.debug(
                f"Activity {row.name}: Number of options after second relaxation: {len(options)}"
            )

        # Attempt 4: if options is still empty (there were no options in possible_zones), return NA
        if options.empty:
            logger.info(f"Activity {row.name}: No options available. Returning NA")
            return "NA"

        # Sample based on "weighting" argument
        if weighting == "floor_area":
            # check the sum of floor_area is not zero
            if options["floor_area"].sum() != 0:
                logger.debug(f"Activity {row.name}: sampling based on floor area")
                selected_zone = options.sample(1, weights="floor_area")[
                    zone_id_col
                ].values[0]
            elif options["counts"].sum() != 0:
                logger.debug(
                    f"Activity {row.name}: No floor area data. sampling based on counts"
                )
                selected_zone = options.sample(1, weights="counts")[zone_id_col].values[
                    0
                ]
            else:
                logger.debug(
                    f"Activity {row.name}: No floor area or count data. sampling randomly"
                )
                selected_zone = options.sample(1)[zone_id_col].values[0]
        elif weighting == "counts":
            if options["counts"].sum() != 0:
                logger.debug(f"Activity {row.name}: sampling based on counts")
                selected_zone = options.sample(1, weights="counts")[zone_id_col].values[
                    0
                ]
            else:
                logger.debug(f"Activity {row.name}: No count data. sampling randomly")
                selected_zone = options.sample(1)[zone_id_col].values[0]
        else:
            logger.debug(f"Activity {row.name}: sampling randomly")
            selected_zone = options.sample(1)[zone_id_col].values[0]

        return selected_zone

    except KeyError:
        logger.info(f"KeyError: Key {row.name} in possible_zones has no values")
        return "NA"


def select_activity(
    row: pd.Series,
    activities_pts: gpd.GeoDataFrame,
    sample_col: str = "none",
) -> pd.Series:
    """
    Select a suitable location for an activity based on the activity purpose and a specific zone
    TODO: this function is specific to education locations. We can either
        - Generalize it to other trip purposes
        - Keep it specific to education and change it''s name
        - Replace it with PAM

    Parameters
    ----------
    row : pandas.Series
        A row from the activity_chains DataFrame
    activities_pts : geopandas.GeoDataFrame
        A GeoDataFrame containing the activities to sample from
    sample_col : str, optional
        The column to sample from, by default 'none'.Options are: "floor_area", "none"


    Returns
    -------
    activity_id : int
        The id of the chosen activity
    activity_geom : shapely.geometry
        The geometry of the chosen activity

    """
    destination_zone = row["dzone"]

    if destination_zone == "NA":
        # log the error
        logger.info(f"Destination zone is NA for row {row}")
        return pd.Series([np.nan, np.nan])

    # filter to activities in the dsired zone
    activities_in_zone = activities_pts[activities_pts["OA21CD"] == destination_zone]

    if activities_in_zone.empty:
        logger.info(f"No activities in zone {destination_zone}")
        return pd.Series([np.nan, np.nan])

    # filter all rows in activities_in_zone where  activities includes the specific activity type
    activities_valid = activities_in_zone[
        activities_in_zone["activities"].apply(lambda x: row["education_type"] in x)
    ]
    # if no activities match the exact education type, relax the constraint to just "education"
    if activities_valid.empty:
        logger.info(
            f"No activities in zone {destination_zone} with education type {row['education_type']},\
                      Returning activities with education type 'education'"
        )
        activities_valid = activities_in_zone[
            activities_in_zone["activities"].apply(lambda x: "education" in x)
        ]
        # if still no activities match the education type, return NA
        if activities_valid.empty:
            logger.info(
                f"No activities in zone {destination_zone} with education type 'education'"
            )
            return pd.Series([np.nan, np.nan])

    if sample_col == "floor_area":
        # sample an activity from activities_valid based on the floor_area column
        if activities_valid["floor_area"].sum() != 0:
            activity = activities_valid.sample(
                1, weights=activities_valid["floor_area"]
            )
        else:
            activity = activities_valid.sample(1)
    else:
        activity = activities_valid.sample(1)

    return pd.Series([activity["id"].values[0], activity["geometry"].values[0]])


def zones_to_time_matrix(
    zones: gpd.GeoDataFrame,
    id_col: str = None,
    to_dict: bool = False,
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


def fill_missing_zones(
    activity: pd.Series,
    travel_times_est: dict,
    activities_per_zone: pd.DataFrame,
    activity_col: str,
    use_mode: bool = False,
):
    """
    This function fills in missing zones in the activity chains. It uses a travel time matrix based on euclidian distance instead of
    the computed travel time matrix which is a sparse matrix.

    Parameters
    ----------
    activity: pd.Series
        A row from the activity chains dataframe
    travel_times_est: dict
        A dictionary with keys as tuples ({id_col}_from, {id_col}_to) and values as dictionaries containing time estimates.
        It is the output of zones_to_time_matrix()
    activities_per_zone: pd.DataFrame
        A dataframe with the number of activities and floorspace for each zone. The columns are 'OA21CD', 'counts', 'floor_area', 'activity'
        where 'activity' is the activity type as defined in the osmox config file
    activity_col: str
        The column name for the activity type
    use_mode: bool
        If True, the function will use the mode of transportation to estimate the travel time: time_{mode}.
        If False, it will use the average travel time: time_average.
        Default is False.

    Returns
    -------
    str
        The zone that has the estimated time closest to the given time. The zone also has an activity that matches the activity_col value.

    """
    activity_purpose = activity[activity_col]
    from_zone = activity["OA21CD"]
    to_zones = activities_per_zone[activities_per_zone["activity"] == activity_purpose][
        "OA21CD"
    ].tolist()

    logger.debug(
        f"Activity {activity.TripID} | person: {activity.id}: Number of possible destination zones: {len(to_zones)}"
    )

    time = activity["TripTotalTime"]
    if use_mode:
        mode = activity["mode"]
    else:
        mode = None

    zone = _get_zones_using_time_estimate(
        estimated_times=travel_times_est,
        from_zone=from_zone,
        to_zones=to_zones,
        time=time,
        mode=mode,
    )

    return zone


def _get_zones_using_time_estimate(
    estimated_times: dict, from_zone: str, to_zones: list, time: int, mode: str = None
) -> str:
    """
    This function returns the zone that has the estimated time closest to the given time. It is meant to be used inside fill_missing_zones()

    Parameters:
    ----------
    estimated_times: dict
        A dictionary with keys as tuples ({id_col}_from, {id_col}_to) and values as dictionaries containing time estimates. It is the output of zones_to_time_matrix()
    id_col: str
        The column name for the zone id.
    from_zone: str
        The zone of the previous activity
    to_zones (list): A list of destination zones.
    time:
        The target time to compare the estimated times with.
    mode: str, optional
        The mode of transportation. It should be one of ['car', 'pt', 'walk', 'cycle']. If not provided, the function uses 'time_average'.

    Returns
    -------
    str
        The zone that has the estimated time closest to the given time.
    """

    acceptable_modes = ["car", "pt", "walk", "cycle"]

    if mode is not None and mode not in acceptable_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Mode must be one of {acceptable_modes}."
        )

    # Convert to_zones to a set for faster lookup
    to_zones_set = set(to_zones)

    # Filter the entries where {id_col}_from matches the specific zone and {id_col}_to is in the specific list of zones
    filtered_dict = {
        k: v
        for k, v in estimated_times.items()
        if k[0] == from_zone and k[1] in to_zones_set
    }
    # get to_zone where time_average is closest to "time"
    if mode is not None:
        closest_to_zone = min(
            filtered_dict.items(), key=lambda item: abs(item[1][f"time_{mode}"] - time)
        )
    else:
        closest_to_zone = min(
            filtered_dict.items(), key=lambda item: abs(item[1]["time_average"] - time)
        )

    return closest_to_zone[0][1]


def filter_matrix_to_boundary(
    boundary,
    matrix,
    boundary_id_col,
    matrix_id_col,
    matrix_id_col_sfx=["_home", "_work"],
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
        The GeoDataFrame containing the zones.

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

    # Create a dictionary where key = zone and values = travel time in minutes
    travel_time_dict = {
        zone: {
            mode: round((dist / speed) / 60, 1)
            for mode, speed in mode_speeds_mps.items()
        }
        for zone, dist in zones["average_dist"].items()
    }

    return travel_time_dict


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

    # Iterate over the keys in the travel_times22 dictionary
    for key in intrazonal_estimates:
        # Create a mask for the rows where 'from_id' and 'to_id' are equal to the current key
        mask = (travel_times_copy["from_id"] == key) & (
            travel_times_copy["to_id"] == key
        )
        # Iterate over the rows that match the mask
        for idx, row in travel_times_copy[mask].iterrows():
            # Replace the 'travel_time_p50' value with the corresponding value from travel_times22
            travel_times_copy.loc[idx, column_to_replace] = intrazonal_estimates[key][
                row["mode"]
            ]

    # Return the modified DataFrame
    return travel_times_copy
