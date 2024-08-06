from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from acbm.logger_config import assigning_primary_locations_logger as logger


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
        error_message = f"Invalid value for weighting: {weighting}. Allowed values are {allowed_weightings}."
        raise ValueError(error_message)

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


def select_facility(
    row: pd.Series,
    facilities_gdf: gpd.GeoDataFrame,
    row_destination_zone_col: str,
    gdf_facility_zone_col: str,
    row_activity_type_col: str,
    gdf_facility_type_col: str,
    fallback_type: Optional[str] = None,
    neighboring_zones: Optional[dict] = None,
    gdf_sample_col: Optional[str] = None,
) -> pd.Series:
    """
    Select a suitable facility based on the activity type and a specific zone from a GeoDataFrame.
    Optionally:
     - looks in neighboring zones when there is no suitable facility in the initial zone
     - add a fallback type to search for a more general type of facility when no specific facilities are found
       (e.g. 'education' instead of 'education_university')
     - sample based on a specific column in the GeoDataFrame (e..g. floor_area)

    Parameters
    ----------
    selection_row : pandas.Series
        A row from the DataFrame indicating the selection criteria, including the destination zone and activity type.
    facilities_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing facilities to sample from.
    row_destination_zone_col : str
        The column name in `selection_row` that indicates the destination zone.
    gdf_facility_zone_col : str
        The column name in `facilities_gdf` that indicates the facility zone.
    row_activity_type_col : str
        The column in `selection_row` indicating the type of activity (e.g., 'education', 'work').
    gdf_facility_type_col : str
        The column in `facilities_gdf` to filter facilities by type based on the activity type.
    fallback_type : Optional[str]
        A more general type of facility to fallback to if no specific facilities are found. By default None.
    neighboring_zones : Optional[dict]
        A dictionary mapping zones to their neighboring zones for fallback searches, by default None.
    gdf_sample_col : Optional[str]
        The column to sample from, by default None. The only feasible input is "floor_area". If "floor_area" is specified,
        uses this column's values as weights for sampling.

    Returns
    -------
    pd.Series
        Series containing the id and geometry of the chosen facility. Returns NaN if no suitable facility is found.
    """
    # Extract the destination zone from the input row
    destination_zone = row[row_destination_zone_col]
    if pd.isna(destination_zone):
        logger.info(f"Destination zone is NA for row {row.name}")
        return pd.Series([np.nan, np.nan])

    # Filter facilities within the specified destination zone
    facilities_in_zone = facilities_gdf[
        facilities_gdf[gdf_facility_zone_col] == destination_zone
    ]
    # Attempt to find facilities matching the specific facility type
    facilities_valid = facilities_in_zone[
        facilities_in_zone[gdf_facility_type_col].apply(
            lambda x: row[row_activity_type_col] in x
        )
    ]

    # If no specific facilities found in the initial zone, and neighboring zones are provided, search in neighboring zones
    if facilities_valid.empty and neighboring_zones:
        logger.info(
            f"No {row[row_activity_type_col]} facilities in {destination_zone}. Expanding search to neighboring zones"
        )
        neighbors = neighboring_zones.get(destination_zone, [])
        facilities_in_neighboring_zones = facilities_gdf[
            facilities_gdf[gdf_facility_zone_col].isin(neighbors)
        ]
        facilities_valid = facilities_in_neighboring_zones[
            facilities_in_neighboring_zones[gdf_facility_type_col].apply(
                lambda x: row[row_activity_type_col] in x
            )
        ]
        logger.info(
            f"Found {len(facilities_valid)} matching facilities in neighboring zones"
        )

    # If no specific facilities found and a fallback type is provided, attempt to find facilities matching the fallback type
    if facilities_valid.empty and fallback_type:
        logger.info(
            f"No {row[row_activity_type_col]} facilities in zone {destination_zone} or neighboring zones, trying with {fallback_type}"
        )
        # This should consider both the initial zone and neighboring zones if the previous step expanded the search
        facilities_valid = facilities_in_zone[
            facilities_in_zone[gdf_facility_type_col].apply(
                lambda x: fallback_type in x
            )
        ]
        logger.info(
            f"Found {len(facilities_valid)} matching facilities with type: {fallback_type}"
        )

    # If no facilities found after all attempts, log the failure and return NaN
    if facilities_valid.empty:
        logger.info(
            f"No facilities in zone {destination_zone} with {gdf_facility_type_col} '{fallback_type or row[row_activity_type_col]}'"
        )
        return pd.Series([np.nan, np.nan])

    # If "floor_area" is specified for sampling
    if (
        gdf_sample_col == "floor_area"
        and "floor_area" in facilities_valid.columns
        and facilities_valid["floor_area"].sum() != 0
    ):
        facility = facilities_valid.sample(1, weights=facilities_valid["floor_area"])
    else:
        # Otherwise, randomly sample one facility from the valid facilities
        facility = facilities_valid.sample(1)

    # Return the id and geometry of the selected facility
    return pd.Series([facility["id"].values[0], facility["geometry"].values[0]])
