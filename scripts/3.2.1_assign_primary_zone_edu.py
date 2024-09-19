import geopandas as gpd
import pandas as pd

import acbm
from acbm.assigning.select_zone_primary import (
    fill_missing_zones,
    select_zone,
)
from acbm.assigning.utils import activity_chains_for_assignment
from acbm.cli import acbm_cli
from acbm.logger_config import assigning_primary_zones_logger as logger
from acbm.preprocessing import add_location
from acbm.utils import Config


@acbm_cli
def main(config_file):
    config = Config(config_file)
    # TODO: consider if RNG seed needs to be distinct for different assignments
    config.init_rng()

    #### LOAD DATA ####

    logger.info("Loading data")

    # --- Possible zones for each activity (calculated in 3.1_assign_possible_zones.py)
    logger.info("Loading possible zones for each activity")
    possible_zones_school = pd.read_pickle(
        acbm.root_path / "data/interim/assigning/possible_zones_education.pkl"
    )

    # --- boundaries
    logger.info("Loading boundaries")

    where_clause = "MSOA21NM LIKE '%Leeds%'"

    boundaries = gpd.read_file(
        acbm.root_path / "data/external/boundaries/oa_england.geojson",
        where=where_clause,
    )

    boundaries = boundaries.to_crs(epsg=4326)

    # --- osm POI data
    logger.info("Loading OSM POI data")

    osm_data_gdf = pd.read_pickle(
        acbm.root_path / "data/interim/assigning/osm_poi_with_zones.pkl"
    )
    # Convert the DataFrame into a GeoDataFrame, and assign a coordinate reference system (CRS)
    logger.info("Converting OSM POI data to GeoDataFrame")

    osm_data_gdf = gpd.GeoDataFrame(osm_data_gdf, geometry="geometry", crs="EPSG:4326")

    # --- Activity chains
    logger.info("Loading activity chains")

    activity_chains = activity_chains_for_assignment()
    activity_chains = activity_chains[activity_chains["TravDay"] == 3]  # Wednesday

    logger.info("Filtering activity chains for trip purpose: education")
    activity_chains_edu = activity_chains[activity_chains["dact"] == "education"]

    logger.info("Assigning activity home locations to boundaries zoning system")

    # Convert location column in activity_chains to spatial column
    centroid_layer = pd.read_csv(
        acbm.root_path / "data/external/centroids/Output_Areas_Dec_2011_PWC_2022.csv"
    )
    activity_chains_edu = add_location(
        activity_chains_edu,
        "EPSG:27700",
        "EPSG:4326",
        centroid_layer,
        "OA11CD",
        "OA11CD",
    )

    # remove index_right column from activity_chains if it exists
    if "index_right" in activity_chains_edu.columns:
        activity_chains_edu = activity_chains_edu.drop(columns="index_right")

    # Spatial join to identify which polygons each point is in
    activity_chains_edu = gpd.sjoin(
        activity_chains_edu,
        boundaries[["OA21CD", "geometry"]],
        how="left",
        predicate="within",
    )
    activity_chains_edu = activity_chains_edu.drop("index_right", axis=1)

    # --- activities per zone
    logger.info("Loading activities per zone")

    activities_per_zone = pd.read_parquet(
        acbm.root_path / "data/interim/assigning/activities_per_zone.parquet"
    )

    # --- travel time estimates
    logger.info("Loading travel time estimates")

    travel_time_estimates = pd.read_pickle(
        acbm.root_path / "data/interim/assigning/travel_time_estimates.pkl"
    )

    #### ASSIGN TO ZONE FROM FEASIBLE ZONES ####

    # For education trips, we use age as an indicator for the type of education facility
    # the individual is most likely to go to. For each person activity, we use the age_group to
    # determine which education facilities to look at.
    #
    # We then sample probabilistically based on the number of facilities in each zone.

    logger.info("Assigning to zone from feasible zones")
    # Apply the function to all rows in activity_chains_example
    activity_chains_edu["dzone"] = activity_chains_edu.apply(
        lambda row: select_zone(
            row=row,
            possible_zones=possible_zones_school,
            activities_per_zone=activities_per_zone,
            id_col="id",
            weighting="floor_area",
            zone_id_col="OA21CD",
        ),
        axis=1,
    )

    mask = activity_chains_edu["dzone"] == "NA"
    # Log the value counts of 'mode' for rows where 'dzone' is "NA"
    mode_counts = activity_chains_edu[mask]["mode"].value_counts()
    logger.info(f"Mode counts for activities with missing zones: {mode_counts}")

    #### FILL IN MISSING ACTIVITIES ####

    # Some activities are not assigned a zone because there is no zone that (a) has the
    # activity, and (b) is reachable using the reprted mode and duration (based on
    # travel_time matrix r5 calculations). For these rows, we fill the zone using times
    # based on euclidian distance and estimated speeds

    logger.info("Filling missing zones")
    # Create a mask for rows where 'dzone' is NaN
    mask = activity_chains_edu["dzone"] == "NA"

    # Apply the function to these rows and assign the result back to 'dzone'
    activity_chains_edu.loc[mask, "dzone"] = activity_chains_edu.loc[mask].apply(
        lambda row: fill_missing_zones(
            activity=row,
            travel_times_est=travel_time_estimates,
            activities_per_zone=activities_per_zone,
            activity_col="education_type",
        ),
        axis=1,
    )

    # Log the value counts of 'mode' for rows where 'dzone' is still "NA" after filling
    mask_after = activity_chains_edu["dzone"] == "NA"
    mode_counts_after = activity_chains_edu[mask_after]["mode"].value_counts()
    logger.info(
        f"Mode counts for activities with missing zones after filling: {mode_counts_after}"
    )

    logger.info("Saving activity chains with assigned zones")

    activity_chains_edu.to_pickle(
        acbm.root_path / "data/interim/assigning/activity_chains_education.pkl"
    )


if __name__ == "__main__":
    main()
