import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen

import acbm
from acbm.assigning.plots import (
    plot_desire_lines,
    plot_scatter_actual_reported,
)
from acbm.assigning.primary_select import (
    fill_missing_zones,
    select_facility,
    select_zone,
)
from acbm.logger_config import assigning_primary_locations_logger as logger
from acbm.preprocessing import add_location

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
    acbm.root_path / "data/external/boundaries/oa_england.geojson", where=where_clause
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

activity_chains = pd.read_parquet(
    acbm.root_path / "data/interim/matching/spc_with_nts_trips.parquet"
)
activity_chains = activity_chains[activity_chains["TravDay"] == 3]  # Wednesday


logger.info("Filtering activity chains for trip purpose: education")
activity_chains_edu = activity_chains[activity_chains["dact"] == "education"]

logger.info("Assigning activity home locations to boundaries zoning system")

# Convert location column in activity_chains to spatial column
centroid_layer = pd.read_csv(
    acbm.root_path / "data/external/centroids/Output_Areas_Dec_2011_PWC_2022.csv"
)
activity_chains_edu = add_location(
    activity_chains_edu, "EPSG:27700", "EPSG:4326", centroid_layer, "OA11CD", "OA11CD"
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


#### ASSING ACTIVITIES TO POINT LOCATIONS ####

logger.info("Assigning activities to point locations")
# 1. Get neighboring zones

# Sometimes, an activity can be assigned to a zone, but there are no facilities
# in the zone that match the activity type. In this case, we can search for matching
# facilities in neighboring zones.

logger.info("Step 1: Getting neighboring zones")

zone_neighbors = Queen.from_dataframe(boundaries, idVariable="OA21CD").neighbors

# 2. select a facility

logger.info("Step 2: Selecting a facility")

# apply the function to a row in activity_chains_ex
activity_chains_edu[["activity_id", "activity_geom"]] = activity_chains_edu.apply(
    lambda row: select_facility(
        row=row,
        facilities_gdf=osm_data_gdf,
        row_destination_zone_col="dzone",
        row_activity_type_col="education_type",
        gdf_facility_zone_col="OA21CD",
        gdf_facility_type_col="activities",
        gdf_sample_col="floor_area",
        neighboring_zones=zone_neighbors,
        fallback_type="education",
    ),
    axis=1,
)

# Compute summary statistics
assignments_successful = activity_chains_edu[
    activity_chains_edu["activity_id"].notnull()
].shape[0]
assignments_failed = activity_chains_edu[
    activity_chains_edu["activity_id"].isnull()
].shape[0]
percentage_failed = (assignments_failed / assignments_successful) * 100

logger.info("Summary statistics for assigning education activities to locations")
logger.info(f"Number of successful assignments: {assignments_successful}")
logger.info(f"Number of failed assignments: {assignments_failed}")
logger.info(f"Percentage of failed assignments: {percentage_failed}")


# save the activity chains as a pickle
logger.info("Saving activity chains with assigned locations")

activity_chains_edu.to_pickle(
    acbm.root_path / "data/interim/assigning/activity_chains_education.pkl"
)


#### PLOTS ####
logger.info("Plotting")

# plot the activity chains
plot_desire_lines(
    activities=activity_chains_edu,
    activity_type_col="dact",
    activity_type="education",
    bin_size=3000,
    boundaries=boundaries,
    sample_size=1000,
    save_dir=acbm.root_path / "data/processed/plots/assigning/",
)

# plot the scatter plot of actual and reported activities
plot_scatter_actual_reported(
    activities=activity_chains_edu,
    x_col="TripTotalTime",
    y_col="length",
    x_label="Reported Travel Time (min)",
    y_label="Actual Distance - Euclidian (km)",
    title_prefix="Scatter plot of TripTotalTime vs. Length",
    activity_type="education",
    save_dir=acbm.root_path / "data/processed/plots/assigning/",
)

plot_scatter_actual_reported(
    activities=activity_chains_edu,
    x_col="TripDisIncSW",
    y_col="length",
    x_label="Reported Travel Distance (km)",
    y_label="Actual Distance - Euclidian (km)",
    title_prefix="Scatter plot of TripDisIncSW vs. Length",
    activity_type="education",
    save_dir=acbm.root_path / "data/processed/plots/assigning/",
)
