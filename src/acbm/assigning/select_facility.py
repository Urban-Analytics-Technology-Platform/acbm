import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from acbm.logger_config import assigning_facility_locations_logger as logger


def _select_facility(
    row: pd.Series,
    unique_id_col: str,
    facilities_gdf: gpd.GeoDataFrame,
    row_destination_zone_col: str,
    gdf_facility_zone_col: str,
    row_activity_type_col: str,
    gdf_facility_type_col: str,
    fallback_type: Optional[str] = None,
    fallback_to_random: bool = False,
    neighboring_zones: Optional[dict] = None,
    gdf_sample_col: Optional[str] = None,
) -> dict:
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
    unique_id_col : str
        The column name in `selection_row` that indicates the unique id. It will be the key of the output dictionary.
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
    fallback_to_random : bool
        If True, sample from all facilities in the zone if no specific facilities are found. By default False.
    neighboring_zones : Optional[dict]
        A dictionary mapping zones to their neighboring zones for fallback searches, by default None.
    gdf_sample_col : Optional[str]
        The column to sample from, by default None. The only feasible input is "floor_area". If "floor_area" is specified,
        uses this column's values as weights for sampling.

    Returns
    -------
    dict
        Dictionary containing the id and geometry of the chosen facility. Returns {'id': np.nan, 'geometry': np.nan} if no suitable facility is found.
    """
    # ----- Step 1. Find valid facilities in the destination zone

    # Extract the destination zone from the input row
    destination_zone = row[row_destination_zone_col]
    if pd.isna(destination_zone):
        logger.info(f"Activity {row.name}: Destination zone is NA")
        return {"id": np.nan, "geometry": np.nan}

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
    logger.info(
        f"Activity {row.name}: Found {len(facilities_valid)} matching facilities in zone {destination_zone}"
    )

    # If no specific facilities found in the initial zone, and neighboring zones are provided, search in neighboring zones
    if facilities_valid.empty and neighboring_zones:
        logger.info(
            f"Activity {row.name}: No {row[row_activity_type_col]} facilities in {destination_zone}. Expanding search to neighboring zones"
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
            f"Activity {row.name}: Found {len(facilities_valid)} matching facilities in neighboring zones"
        )

    # If no specific facilities found and a fallback type is provided, attempt to find facilities matching the fallback type
    if facilities_valid.empty and fallback_type:
        logger.info(
            f"Activity {row.name}: No {row[row_activity_type_col]} facilities in zone {destination_zone} or neighboring zones, trying with {fallback_type}"
        )
        # This should consider both the initial zone and neighboring zones if the previous step expanded the search
        facilities_valid = facilities_in_zone[
            facilities_in_zone[gdf_facility_type_col].apply(
                lambda x: fallback_type in x
            )
        ]
        logger.info(
            f"Activity {row.name}: Found {len(facilities_valid)} matching facilities with type: {fallback_type}"
        )

    # if no specific facilities found and fallback_to_random is True, take all facilities in the zone
    if facilities_valid.empty and fallback_to_random:
        logger.info(
            f"Activity {row.name}: No facilities in zone {destination_zone} with {gdf_facility_type_col} '{fallback_type or row[row_activity_type_col]}'. Sampling from all facilities in the zone"
        )
        facilities_valid = facilities_in_zone

    # If no facilities found after all attempts, log the failure and return NaN
    if facilities_valid.empty:
        logger.info(
            f"Activity {row.name}: No facilities in zone {destination_zone} with {gdf_facility_type_col} '{fallback_type or row[row_activity_type_col]}'"
        )
        return {row[unique_id_col]: (np.nan, np.nan)}

    # ----- Step 2. Sample a facility from the valid facilities

    # If "floor_area" is specified for sampling
    if (
        gdf_sample_col == "floor_area"
        and "floor_area" in facilities_valid.columns
        and facilities_valid["floor_area"].sum() != 0
    ):
        # Ensure floor_area is numeric
        facilities_valid["floor_area"] = pd.to_numeric(
            facilities_valid["floor_area"], errors="coerce"
        )
        facilities_valid = facilities_valid.dropna(subset=["floor_area"])
        facility = facilities_valid.sample(1, weights=facilities_valid["floor_area"])
        logger.info(f"Activity {row.name}: Sampled facility based on floor area)")
    else:
        # Otherwise, randomly sample one facility from the valid facilities
        facility = facilities_valid.sample(1)
        logger.info(f"Activity {row.name}: Sampled facility randomly")

    # Return the id and geometry of the selected facility
    return {
        row[unique_id_col]: (facility["id"].values[0], facility["geometry"].values[0])
    }


def select_facility(
    df: pd.DataFrame,
    unique_id_col: str,
    facilities_gdf: gpd.GeoDataFrame,
    row_destination_zone_col: str,
    gdf_facility_zone_col: str,
    row_activity_type_col: str,
    gdf_facility_type_col: str,
    gdf_sample_col: Optional[str] = None,
    neighboring_zones: Optional[dict] = None,
    fallback_type: Optional[str] = None,
    fallback_to_random: bool = False,
) -> pd.DataFrame:
    """
    Select facilities for each row in the DataFrame based on the provided logic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the activity chains.
    facilities_gdf : gpd.GeoDataFrame
        GeoDataFrame containing facilities to sample from.
    row_destination_zone_col : str
        The column name in `df` that indicates the destination zone.
    gdf_facility_zone_col : str
        The column name in `facilities_gdf` that indicates the facility zone.
    row_activity_type_col : str
        The column in `df` indicating the type of activity (e.g., 'education', 'work').
    gdf_facility_type_col : str
        The column in `facilities_gdf` to filter facilities by type based on the activity type.
    gdf_sample_col : Optional[str]
        The column to sample from, by default None. The only feasible input is "floor_area".
    neighboring_zones : Optional[dict]
        A dictionary mapping zones to their neighboring zones for fallback searches, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the selected facilities for each row.
    """
    # Initialize a dictionary to store the selected facilities
    selected_facilities = {}

    # Select a facility for each row in the DataFrame
    for index, row in df.iterrows():
        selected_facility = _select_facility(
            row=row,
            unique_id_col=unique_id_col,
            facilities_gdf=facilities_gdf,
            row_destination_zone_col=row_destination_zone_col,
            row_activity_type_col=row_activity_type_col,
            gdf_facility_zone_col=gdf_facility_zone_col,
            gdf_facility_type_col=gdf_facility_type_col,
            gdf_sample_col=gdf_sample_col,
            neighboring_zones=neighboring_zones,
            fallback_type=fallback_type,
            fallback_to_random=fallback_to_random,
        )

        # Update the dictionary with the selected facility
        selected_facilities.update(selected_facility)

    return selected_facilities


def map_activity_locations(
    activity_chains_df: pd.DataFrame, activity_locations_dict: dict, id_col: str = "pid"
):
    """
    Map activity locations to the activity chains DataFrame.

    Parameters
    ----------
    activity_chains_df : pd.DataFrame
        DataFrame containing the activity chains.
    activity_locations_dict : dict
        Dictionary containing the activity locations.
    pid_col : str, optional
        The column name in `activity_chains_df` that contains the unique identifiers, by default 'pid'.

    Returns
    -------
    pd.DataFrame
        DataFrame with mapped activity locations.
    """
    activity_chains_df["end_location_id"] = activity_chains_df[id_col].map(
        lambda pid: activity_locations_dict[pid][0]
        if pid in activity_locations_dict
        else None
    )
    activity_chains_df["end_location_geometry"] = activity_chains_df[id_col].map(
        lambda pid: activity_locations_dict[pid][1]
        if pid in activity_locations_dict
        else None
    )
    return activity_chains_df
