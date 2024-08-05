import pickle as pkl

import geopandas as gpd
import pandas as pd

import acbm
from acbm.assigning.assigning import (
    filter_matrix_to_boundary,
    get_activities_per_zone,
    get_possible_zones,
    intrazone_time,
    replace_intrazonal_travel_time,
    zones_to_time_matrix,
)
from acbm.logger_config import assigning_primary_logger as logger
from acbm.preprocessing import add_location

#### LOAD DATA ####

# --- Activity chains
logger.info("Loading activity chains")
activity_chains = pd.read_parquet("../data/interim/matching/spc_with_nts_trips.parquet")
logger.info("Activity chains loaded")

# Filter to a specific day of the week
logger.info("Filtering activity chains to a specific day of the week")
activity_chains = activity_chains[activity_chains["TravDay"] == 3]  # Wednesday

# --- Study area boundaries

logger.info("Loading study area boundaries")
where_clause = "MSOA21NM LIKE '%Leeds%'"

boundaries = gpd.read_file(
    acbm.root_path / "data/external/boundaries/oa_england.geojson", where=where_clause
)

# convert boundaries to 4326
boundaries = boundaries.to_crs(epsg=4326)

logger.info("Study area boundaries loaded")

# --- Assign activity home locations to boundaries zoning system

logger.info("Assigning activity home locations to boundaries zoning system")
# Convert location column in activity_chains to spatial column
centroid_layer = pd.read_csv(
    acbm.root_path / "data/external/centroids/Output_Areas_Dec_2011_PWC_2022.csv"
)
activity_chains = add_location(
    activity_chains, "EPSG:27700", "EPSG:4326", centroid_layer, "OA11CD", "OA11CD"
)

# Convert the DataFrame into a GeoDataFrame, and assign a coordinate reference system (CRS)
activity_chains = gpd.GeoDataFrame(activity_chains, geometry="location")
activity_chains.crs = "EPSG:4326"  # I assume this is the crs


# remove index_right column from activity_chains if it exists
if "index_right" in activity_chains.columns:
    activity_chains = activity_chains.drop(columns="index_right")


# Spatial join to identify which polygons each point is in
activity_chains = gpd.sjoin(
    activity_chains, boundaries[["OA21CD", "geometry"]], how="left", predicate="within"
)
activity_chains = activity_chains.drop("index_right", axis=1)


# --- Travel time matrix for study area
#
# Travel time data between geographical areas (LSOA, OA, custom hexagons etc) is used
# to determine feasible work / school locations for each individual. The travel times
# are compared to the travel times of the individual's actual trips from the nts
# (`tst`/`TripStart` and `tet`/`TripEnd`)

logger.info("Loading travel time matrix")

travel_times = pd.read_parquet(
    acbm.root_path / "data/external/travel_times/oa/travel_time_matrix_acbm.parquet"
)

logger.info("Travel time matrix loaded")

logger.info("Merging travel time matrix with boundaries")

# convert from_id and to_id to int to match the boundaries data type
travel_times = travel_times.astype({"from_id": int, "to_id": int})

# merge travel_times with boundaries
travel_times = travel_times.merge(
    boundaries[["OBJECTID", "OA21CD"]],
    left_on="from_id",
    right_on="OBJECTID",
    how="left",
)
travel_times = travel_times.drop(columns="OBJECTID")

travel_times = travel_times.merge(
    boundaries[["OBJECTID", "OA21CD"]],
    left_on="to_id",
    right_on="OBJECTID",
    how="left",
    suffixes=("_from", "_to"),
)
travel_times = travel_times.drop(columns="OBJECTID")

# #### Travel distance matrix
#
# Some areas aren't reachable by specific modes. We create a travel distance matrix
# to fall back on when there are no travel time calculations

logger.info("Creating travel time estimates")

travel_time_estimates = zones_to_time_matrix(
    zones=boundaries, id_col="OA21CD", to_dict=True
)

with open(
    acbm.root_path / "data/interim/assigning/travel_time_estimates.pkl", "wb"
) as f:
    pkl.dump(travel_time_estimates, f)

logger.info("Travel time estimates created")

# --- Intrazonal trip times
#
# Intrazonal trips all have time = 0. Our `get_possible_zones` function finds zones
# that are within a specified % threshold from the reported time in the NTS.
# A threshold percentage from a non zero number never equals 0, so intrazonal trips
# are not found. The problem is also explained in this issue #30
#
# Below, we assign intrazonal trips a non-zero time based on the zone area

# get intrazone travel time estimates per mode

logger.info("Creating intrazonal travel time estimates")

intrazone_times = intrazone_time(boundaries)

logger.info("Intrazonal travel time estimates created")

# replace intrazonal travel times with estimates from intrazone_times
logger.info("Replacing intrazonal travel times with estimates from intrazone_times")

travel_times = replace_intrazonal_travel_time(
    travel_times=travel_times,
    intrazonal_estimates=intrazone_times,
    column_to_replace="travel_time_p50",
)

logger.info("Intrazonal travel times replaced")


# --- Activity locations (Facilities)
#
# Activity locations are obtained from OSM using the [osmox](https://github.com/arup-group/osmox)
# package. Check the config documentation in the package and the `config_osmox` file in this repo


logger.info("Loading activity locations")

# osm data
osm_data = gpd.read_parquet(
    acbm.root_path / "data/external/boundaries/west-yorkshire_epsg_4326.parquet"
)

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
# spatial join to identify which zone each point in osm_data is in
osm_data_gdf = gpd.sjoin(
    osm_data, boundaries[["OA21CD", "geometry"]], how="inner", predicate="within"
)

activities_per_zone = get_activities_per_zone(
    zones=boundaries, zone_id_col="OA21CD", activity_pts=osm_data, return_df=True
)


# --- Commuting matrices (from 2021 census)

commute_level = "OA"  # "OA" or "MSOA" data

logger.info(f"Loading commuting matrices at {commute_level} level")

# Clean the data

if commute_level == "MSOA":
    print("Step 1: Reading in the zipped csv file")
    travel_demand = pd.read_csv(acbm.root_path / "data/external/ODWP15EW_MSOA_v1.zip")

    print("Step 2: Creating commute_mode_dict")
    commute_mode_dict = {
        "Bus, minibus or coach": "pt",
        "Driving a car or van": "car",
        "Train": "pt",
        "Underground, metro, light rail, tram": "pt",
        "On foot": "walk",
        "Taxi": "car",
        "Other method of travel to work": "other",
        "Bicycle": "cycle",
        "Passenger in a car or van": "car",
        "Motorcycle, scooter or moped": "car",
        "Work mainly at or from home": "home",
    }

    print("Step 3: Mapping commute mode to model mode")
    travel_demand["mode"] = travel_demand[
        "Method used to travel to workplace (12 categories) label"
    ].map(commute_mode_dict)

    print("Step 4: Filtering rows and dropping unnecessary columns")
    travel_demand_clipped = travel_demand[
        travel_demand["Place of work indicator (4 categories) code"].isin([1, 3])
    ]
    travel_demand_clipped = travel_demand_clipped.drop(
        columns=[
            "Middle layer Super Output Areas label",
            "MSOA of workplace label",
            "Method used to travel to workplace (12 categories) label",
            "Method used to travel to workplace (12 categories) code",
            "Place of work indicator (4 categories) code",
            "Place of work indicator (4 categories) label",
        ]
    )

    print("Step 5: Renaming columns and grouping")
    travel_demand_clipped = travel_demand_clipped.rename(
        columns={
            "Middle layer Super Output Areas code": "MSOA21CD_home",
            "MSOA of workplace code": "MSOA21CD_work",
        }
    )
    travel_demand_clipped = (
        travel_demand_clipped.groupby(["MSOA21CD_home", "MSOA21CD_work", "mode"])
        .agg({"Count": "sum"})
        .reset_index()
    )

    print("Step 6: Filtering matrix to boundary")
    travel_demand_clipped = filter_matrix_to_boundary(
        boundary=boundaries,
        matrix=travel_demand_clipped,
        boundary_id_col="MSOA21CD",
        matrix_id_col="MSOA21CD",
        type="both",
    )

elif commute_level == "OA":
    print("Step 1: Reading in the zipped csv file")
    travel_demand = pd.read_csv(acbm.root_path / "data/external/ODWP01EW_OA.zip")

    print("Step 2: Filtering rows and dropping unnecessary columns")
    travel_demand_clipped = travel_demand[
        travel_demand["Place of work indicator (4 categories) code"].isin([1, 3])
    ]
    travel_demand_clipped = travel_demand_clipped.drop(
        columns=[
            "Place of work indicator (4 categories) code",
            "Place of work indicator (4 categories) label",
        ]
    )

    print("Step 3: Renaming columns and grouping")
    travel_demand_clipped = travel_demand_clipped.rename(
        columns={
            "Output Areas code": "OA21CD_home",
            "OA of workplace code": "OA21CD_work",
        }
    )
    travel_demand_clipped = (
        travel_demand_clipped.groupby(["OA21CD_home", "OA21CD_work"])
        .agg({"Count": "sum"})
        .reset_index()
    )

    print("Step 4: Filtering matrix to boundary")
    travel_demand_clipped = filter_matrix_to_boundary(
        boundary=boundaries,
        matrix=travel_demand_clipped,
        boundary_id_col="OA21CD",
        matrix_id_col="OA21CD",
        type="both",
    )

logger.info(f"Commuting matrices at {commute_level} level loaded")

# Get dictionary of commuting matrices
logger.info("Converting commuting matrices to dictionaries")

if commute_level == "MSOA":
    travel_demand_dict_mode = (
        travel_demand_clipped.groupby(["MSOA21CD_home", "MSOA21CD_work"])
        .apply(lambda x: dict(zip(x["mode"], x["Count"])))
        .to_dict()
    )
    travel_demand_dict_nomode = (
        travel_demand_clipped.groupby(["MSOA21CD_home", "MSOA21CD_work"])["Count"]
        .sum()
        .to_dict()
    )

elif commute_level == "OA":
    travel_demand_dict_nomode = (
        travel_demand_clipped.groupby(["OA21CD_home", "OA21CD_work"])["Count"]
        .sum()
        .to_dict()
    )

logger.info("Commuting matrices converted to dictionaries")


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
    filter_by_activity=True,
    activity_col="education_type",
    time_tolerance=0.2,
)

logger.info("Saving feasible zones for education activities")
# save possible_zones_school to dictionary
with open(
    acbm.root_path / "data/interim/assigning/possible_zones_education.pkl", "wb"
) as f:
    pkl.dump(possible_zones_school, f)

del possible_zones_school

logger.info("Filtering activity chains to only include work activities")

activity_chains_work = activity_chains[activity_chains["dact"] == "work"]

logger.info("Getting feasible zones for each work activity")

possible_zones_work = get_possible_zones(
    activity_chains=activity_chains_work,
    travel_times=travel_times,
    activities_per_zone=activities_per_zone,
    filter_by_activity=True,
    activity_col="dact",
    time_tolerance=0.2,
)

logger.info("Saving feasible zones for work activities")

# save possible_zones_work to dictionary
with open(acbm.root_path / "data/interim/assigning/possible_zones_work.pkl", "wb") as f:
    pkl.dump(possible_zones_work, f)

del possible_zones_work
