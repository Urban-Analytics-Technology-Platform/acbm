import geopandas as gpd
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


def _map_time_to_day_part(minutes: int, time_of_day_mapping: dict = None) -> str:
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
    elif day in [6, 7]:
        return 0
    else:
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


def get_possible_zones_old(
    activity: pd.Series, travel_times: pd.DataFrame, time_tolerance: int = 0.2
) -> dict:
    """
    Get possible zones for a given activity chain. It is applied on a single row from the activity dataframe.

    Parameters
    ----------
    activity: pd.Series
        A row from the activity chains dataframe. It should contain the following columns: 'tst', 'TripTotalTime', 'mode', 'OA21CD'
    travel_times: pd.DataFrame
        A dataframe with travel times between zones
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
    # get the mode
    mode = activity["mode"]

    # get the origin zone
    origin_zone = activity["OA21CD"]

    # filter the travel_times dataframe by trip_origin and mode
    travel_times_filtered_origin_mode = travel_times[
        (travel_times["OA21CD_from"] == origin_zone)
        & (travel_times["combination"].apply(lambda x: x.split("_")[0]) == mode)
    ]

    # if the trip is being done by pt, we need to filter the travel_times data based on time_of_day and weekday/weekend
    if mode == "pt":
        travel_times_filtered_origin_mode = get_travel_times_pt(
            activity, travel_times_filtered_origin_mode
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

    # if travel_times_filtered_time returns an empty df, select the row with the closest time to the reported time
    if travel_times_filtered_time.empty:
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


def get_possible_zones(
    activity_chains: pd.DataFrame,
    travel_times: pd.DataFrame,
    # list_of_times_of_day: list = ["morning", "afternoon", "evening", "night"],
    time_tolerance: int = 0.2,
) -> dict:
    """
    Get possible zones for all activity chains in the dataset. This function loops over the travel_times dataframe and filters by mode, time of day and weekday/weekend.
    At each loop it applies the get_possible_zones function to each row in the activity_chains dataframe.
    The travel_times dataframe is big, so doing some initial filtering before applying _get_possible_zones makes the process faster as less filtering is done for each
    row when running _get_possible_zones.

    Parameters
    ----------
    activity_chains: pd.DataFrame
        A dataframe with activity chains
    travel_times: pd.DataFrame
        A dataframe with travel times between zones
    list_of_times_of_day: list
        A list of times of day to consider
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
                            lambda row: _get_possible_zones(
                                activity=row,
                                travel_times=travel_times_filtered_mode_time_day,
                                time_tolerance=0.1,
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
                # apply get_possible_zones to each row in activity_chains_filtered
                # pandarallel.initialize(progress_bar=True)
                possible_zones = activity_chains_filtered.parallel_apply(
                    lambda row: _get_possible_zones(
                        activity=row,
                        travel_times=travel_times_filtered_mode_time_day,
                        time_tolerance=0.1,
                    ),
                    axis=1,
                )

                results.update(possible_zones)

    return results


def _get_possible_zones(
    activity: pd.Series, travel_times: pd.DataFrame, time_tolerance: int = 0.2
) -> dict:
    """
    Get possible zones for a given activity chain

    Parameters
    ----------
    activity: pd.Series
        A row from the activity chains dataframe. It should contain the following columns: 'tst', 'TripTotalTime', 'mode', 'OA21CD'
    travel_times: pd.DataFrame
        A dataframe with travel times between zones
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

    # filter the travel_times dataframe by trip_origin and mode
    travel_times_filtered_origin_mode = travel_times[
        travel_times["OA21CD_from"] == origin_zone
    ]

    # if the trip is being done by pt, we need to filter the travel_times data based on time_of_day and weekday/weekend
    # if mode == "pt":
    #     travel_times_filtered_origin_mode = get_travel_times_pt(activity,
    #                                                             travel_times_filtered_origin_mode)

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

    # if travel_times_filtered_time returns an empty df, select the row with the closest time to the reported time
    if travel_times_filtered_time.empty:
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
    zones: gpd.GeoDataFrame, zone_id_col: str, activity_pts: gpd.GeoDataFrame
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

    return grouped_data
