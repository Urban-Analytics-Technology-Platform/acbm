import pandas as pd

def map_time_to_day_part(minutes: int, time_of_day_mapping: dict = None) -> str:
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
        range(0, 300): 'night',       # 00:00 - 04:59
        range(300, 720): 'morning', # 05:00 - 11:59
        range(720, 1080): 'afternoon', # 12:00 - 17:59
        range(1080, 1440): 'evening'  # 18:00 - 23:59
    }

    # use the default dictionary if none is provided
    if time_of_day_mapping is None:
        time_of_day_mapping = default_time_of_day_mapping
    
    for time_range, description in time_of_day_mapping.items():
        if minutes in time_range:
            return description
    return 'unknown'  # return 'unknown' if the time doesn't fall into any of the defined ranges


def map_day_to_wkday_binary(day: int) -> int:
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
    return 0




# function to filter travel_times df specifically for pt mode
# match time_of_day to pt option

def get_travel_times_pt(activity: pd.Series,
                        travel_times: pd.DataFrame
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
    activity['time_of_day'] = map_time_to_day_part(activity['tst'])
    # use map_day_to_wkday_binary to identify if activity is on a weekday or weekend 
    activity['weekday'] = map_day_to_wkday_binary(activity['TravDay'])

    # if weekday = 1, filter travel_times df to rows where combination contains pt_wkday and the time_of_day
    if activity['weekday'] == 1:
        travel_times_filtered = travel_times[
            (travel_times["combination"].str.contains('pt_wkday')) &
            (travel_times["combination"].str.contains(activity['time_of_day']))
        ]
    # if weekday = 0, filter travel_times df to rows where combination contains pt_wkend 
    # TODO: get travel_times for all times during the weekend (morning, afternoon, evening, night) 
    # and apply the same logic as weekday
    else:
        travel_times_filtered = travel_times[
            (travel_times["combination"].str.contains('pt_wkend'))
        ]

    return travel_times_filtered

    # TODO: if travel_times_filtered is empty, select the zones that are within the reported distance "TripDisIncSW"
 


def get_possible_zones(activity: pd.Series, 
                       travel_times: pd.DataFrame, 
                       time_tolerance: int = 0.2) -> dict:
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

    # get the time of day
    time_of_day = map_time_to_day_part(activity['tst'])
    # get the travel time
    travel_time = activity['TripTotalTime']
    # get the mode
    mode = activity['mode']

    # get the origin zone
    origin_zone = activity['OA21CD']

    # filter the travel_times dataframe by trip_origin and mode
    travel_times_filtered_origin_mode = travel_times[
        (travel_times["OA21CD_from"] == origin_zone) & 
        (travel_times["combination"].apply(lambda x: x.split('_')[0]) == mode)
    ]

    # if the trip is being done by pt, we need to filter the travel_times data based on time_of_day and weekday/weekend
    if mode == "pt":
        travel_times_filtered_origin_mode = get_travel_times_pt(activity, 
                                                                travel_times_filtered_origin_mode)


    # filter by reported trip time
    travel_times_filtered_time = travel_times_filtered_origin_mode[
        (travel_times_filtered_origin_mode["travel_time_p50"] >= travel_time - time_tolerance*travel_time) & 
        (travel_times_filtered_origin_mode["travel_time_p50"] <= travel_time + time_tolerance*travel_time)
    ]

    # if travel_times_filtered_time returns an empty df, select the row with the closest time to the reported time
    if travel_times_filtered_time.empty:
        travel_times_filtered_time = travel_times_filtered_origin_mode.iloc[(travel_times_filtered_origin_mode['travel_time_p50'] - travel_time).abs().argsort()[:1]]

    # create dictionary with key = origin_zone and values = list of travel_times_filtered.OA21CD_to
    possible_zones = travel_times_filtered_time.groupby('OA21CD_from')['OA21CD_to'].apply(list).to_dict()

    return possible_zones 
