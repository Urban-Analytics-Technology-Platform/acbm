import pickle as pkl

import geopandas as gpd
import pandas as pd

from acbm.assigning.feasible_zones_primary import get_possible_zones
from acbm.assigning.utils import (
    activity_chains_for_assignment,
    get_activities_per_zone,
    get_chosen_day,
    intrazone_time,
    replace_intrazonal_travel_time,
    zones_to_time_matrix,
)
from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config
from acbm.preprocessing import add_locations_to_activity_chains


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("assigning_primary_feasible", __file__)

    #### LOAD DATA ####

    # --- Activity chains
    logger.info("Loading activity chains")
    activity_chains = activity_chains_for_assignment(config)
    logger.info("Activity chains loaded")

    logger.info("Filtering activity chains to a specific day of the week")

    # Generate random sample of days by household
    get_chosen_day(activity_chains, config.parameters.common_household_day).to_parquet(
        config.output_path / "interim" / "assigning" / "chosen_trav_day.parquet"
    )

    # Filter to chosen day
    activity_chains = activity_chains_for_assignment(config, subset_to_chosen_day=True)

    # --- Study area boundaries

    logger.info("Loading study area boundaries")
    boundaries = config.get_study_area_boundaries()
    logger.info(f"Study area boundaries loaded and reprojected to {config.output_crs}")

    # --- Assign activity home locations to boundaries zoning system

    logger.info("Assigning activity home locations to boundaries zoning system")

    # add home location (based on OA11CD from SPC)
    activity_chains = add_locations_to_activity_chains(
        activity_chains=activity_chains,
        target_crs=f"EPSG:{config.output_crs}",
        centroid_layer=pd.read_csv(config.centroid_layer_filepath),
    )

    # remove index_right column from activity_chains if it exists
    if "index_right" in activity_chains.columns:
        activity_chains = activity_chains.drop(columns="index_right")

    # Spatial join to identify which polygons each point is in
    activity_chains = gpd.sjoin(
        activity_chains,
        boundaries[[config.zone_id, "geometry"]],
        how="left",
        predicate="within",
    )
    activity_chains = activity_chains.drop("index_right", axis=1)

    # --- Travel time matrix for study area
    #
    # Travel time data between geographical areas (LSOA, OA, custom hexagons etc) is used
    # to determine feasible work / school locations for each individual. The travel times
    # are compared to the travel times of the individual's actual trips from the nts
    # (`tst`/`TripStart` and `tet`/`TripEnd`)

    logger.info("Creating estimated travel times matrix")
    # Create a new travel time matrix based on distances between zones
    travel_time_estimates = zones_to_time_matrix(
        zones=boundaries,
        id_col=config.zone_id,
        time_units="m",
        detour_factor=config.feasible_assignment.detour_factor,
        decay_rate=config.feasible_assignment.decay_rate,
    )
    logger.info("Travel time estimates created")

    if config.parameters.travel_times:
        logger.info("Loading travel time matrix")
        try:
            travel_times = pd.read_parquet(config.travel_times_filepath)
            print("Travel time matrix loaded successfully.")
        except Exception as e:
            logger.info(
                f"Failed to load travel time matrix: {e}. Check that you have a "
                "travel_times matrix at {travel_time_matrix_path}. Otherwise set "
                "travel_times to false in config"
            )
            raise e
    else:
        # If travel_times is not true, set travel_times as travel_times_estimates
        travel_times = travel_time_estimates

    # --- Intrazonal trip times
    #
    # Intrazonal trips all have time = 0. Our `get_possible_zones` function finds zones
    # that are within a specified % threshold from the reported time in the NTS.
    # A threshold percentage from a non zero number never equals 0, so intrazonal trips
    # are not found. The problem is also explained in this issue #30

    # Below, we assign intrazonal trips a non-zero time based on the zone area

    # get intrazone travel time estimates per mode

    logger.info("Creating intrazonal travel time estimates")

    intrazone_times = intrazone_time(zones=boundaries, key_column=config.zone_id)

    # save intrazone_times to pickle
    with open(config.interim_path / "assigning" / "intrazone_times.pkl", "wb") as f:
        pkl.dump(intrazone_times, f)

    logger.info("Intrazonal travel time estimates created")

    # replace intrazonal travel times with estimates from intrazone_times
    logger.info("Replacing intrazonal travel times with estimates from intrazone_times")

    travel_times = replace_intrazonal_travel_time(
        travel_times=travel_times,
        intrazonal_estimates=intrazone_times,
        column_to_replace="time",
    )

    logger.info("Intrazonal travel times replaced")

    # save travel_time_etstimates as parquet
    travel_times.to_parquet(config.travel_times_estimates_filepath)

    # --- Activity locations (Facilities)
    #
    # Activity locations are obtained from OSM using the [osmox](https://github.com/arup-group/osmox)
    # package. Check the config documentation in the package and the `config_osmox` file in this repo

    logger.info("Loading activity locations")

    # osm data
    osm_data = gpd.read_parquet(config.osm_path)

    logger.info("Activity locations loaded")
    # remove rows with activities = home OR transit

    osm_data = osm_data[~osm_data["activities"].isin(["home", "transit"])]

    # --- Get the number of activities in each zone
    #
    # Each zone has a different number of education facilities.
    # We can use the number of facilities in each zone to determine the probability
    # of each zone being chosen for each trip. We can then use these probabilities to
    # randomly assign a zone to each trip.

    logger.info("Getting the number of activities in each zone")

    # ensure that osm_data_gdf and boundaries are in the same crs
    osm_data = osm_data.to_crs(boundaries.crs)

    # spatial join to identify which zone each point in osm_data is in
    osm_data_gdf = gpd.sjoin(
        osm_data,
        boundaries[[config.zone_id, "geometry"]],
        how="inner",
        predicate="within",
    )
    # save as pickle
    osm_data_gdf.to_pickle(config.osm_poi_with_zones)

    activities_per_zone = get_activities_per_zone(
        zones=boundaries,
        zone_id_col=config.zone_id,
        activity_pts=osm_data,
        return_df=True,
    )

    activities_per_zone.to_parquet(config.activities_per_zone)

    #### Get possible zones for each primary activity

    # # For education trips, we use age as an indicator for the type of education facility the individual is most likely to go to. The `age_group_mapping` dictionary maps age groups to education facility types. For each person activity, we use the age_group to determine which education facilities to look at.

    # map the age_group to an education type (age group is from NTS::Age_B04ID)
    # TODO check if age_group_mapping done properly in script 2,
    # then move it even further upstream

    # filter activity_chains to only include primary activities

    logger.info("Filtering activity chains to only include education activities")

    activity_chains_edu = activity_chains[activity_chains["dact"] == "education"]

    logger.info("Getting feasible zones for each education activity")

    possible_zones_school = get_possible_zones(
        activity_chains=activity_chains_edu,
        travel_times=travel_times,
        activities_per_zone=activities_per_zone,
        boundaries=boundaries,
        key_col="id",
        zone_id=config.zone_id,
        filter_by_activity=True,
        activity_col="education_type",
        time_tolerance=config.parameters.tolerance_edu
        if config.parameters.tolerance_edu is not None
        else 0.3,
        detour_factor=config.feasible_assignment.detour_factor,
        decay_rate=config.feasible_assignment.decay_rate,
    )

    logger.info("Saving feasible zones for education activities")
    # save possible_zones_school to dictionary
    with open(config.possible_zones_education, "wb") as f:
        pkl.dump(possible_zones_school, f)

    del possible_zones_school

    logger.info("Filtering activity chains to only include work activities")

    activity_chains_work = activity_chains[activity_chains["dact"] == "work"]

    logger.info("Getting feasible zones for each work activity")

    possible_zones_work = get_possible_zones(
        activity_chains=activity_chains_work,
        travel_times=travel_times,
        activities_per_zone=activities_per_zone,
        boundaries=boundaries,
        key_col="id",
        zone_id=config.zone_id,
        filter_by_activity=True,
        activity_col="dact",
        time_tolerance=config.parameters.tolerance_work
        if config.parameters.tolerance_work is not None
        else 0.3,
        detour_factor=config.feasible_assignment.detour_factor,
        decay_rate=config.feasible_assignment.decay_rate,
    )

    logger.info("Saving feasible zones for work activities")

    # save possible_zones_work to dictionary
    with open(config.possible_zones_work, "wb") as f:
        pkl.dump(possible_zones_work, f)

    del possible_zones_work


if __name__ == "__main__":
    main()
