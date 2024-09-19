import pandas as pd
from pandarallel import pandarallel

from acbm.assigning.utils import _map_day_to_wkday_binary, _map_time_to_day_part
from acbm.logger_config import assigning_primary_feasible_logger as logger
from acbm.utils import Config

pandarallel.initialize(progress_bar=True)


def get_possible_zones(
    activity_chains: pd.DataFrame,
    travel_times: pd.DataFrame,
    activities_per_zone: pd.DataFrame,
    activity_col: str,
    key_col: str,
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
    key_col: str
        The column that will be used as a key in the dictionary
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
        with the origin zone as the key and a list of possible destination zones as the value. Eg:
        {
        164: {'E00059011': ['E00056917','E00056922', 'E00056923']},
        165: {'E00059012': ['E00056918','E00056952', 'E00056923']}
        }
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

    # Initialize a list to collect results
    results_list = []

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
                            lambda row, tt=travel_times_filtered_mode_time_day: {
                                row[key_col]: _get_possible_zones(
                                    activity=row,
                                    travel_times=tt,
                                    activities_per_zone=activities_per_zone,
                                    filter_by_activity=filter_by_activity,
                                    activity_col=activity_col,
                                    time_tolerance=time_tolerance,
                                )
                            },
                            axis=1,
                        )

                        results_list.extend(possible_zones)

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
                    lambda row, tt=travel_times_filtered_mode_time_day: {
                        row[key_col]: _get_possible_zones(
                            activity=row,
                            travel_times=tt,
                            activities_per_zone=activities_per_zone,
                            filter_by_activity=filter_by_activity,
                            activity_col=activity_col,
                            time_tolerance=time_tolerance,
                        )
                    },
                    axis=1,
                )

                results_list.extend(possible_zones)

    # Combine all dictionaries in the list into a single dictionary
    results = {}
    for result in results_list:
        for key, value in result.items():
            results[key] = value

    return results


def _get_possible_zones(
    activity: pd.Series,
    travel_times: pd.DataFrame,
    activities_per_zone: pd.DataFrame,
    filter_by_activity: bool,
    activity_col: str,
    zone_id: str,
    time_tolerance: float = 0.2,
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
    origin_zone = activity[zone_id]
    # get the activity purpose
    activity_purpose = activity[activity_col]

    # filter the travel_times dataframe by trip_origin and activity_purpose
    travel_times_filtered_origin_mode = travel_times[
        travel_times[Config.get_origin_zone_id(zone_id)] == origin_zone
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
            travel_times_filtered_origin_mode[
                Config.get_destination_zone_id(zone_id)
            ].isin(filtered_activities_per_zone[zone_id])
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
    return (
        travel_times_filtered_time.groupby(Config.get_origin_zone_id(zone_id))[
            Config.get_destination_zone_id(zone_id)
        ]
        .apply(list)
        .to_dict()
    )
