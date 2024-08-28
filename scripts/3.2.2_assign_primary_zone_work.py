import os

import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen

import acbm
from acbm.assigning.plots import (
    plot_desire_lines,
    plot_scatter_actual_reported,
    plot_workzone_assignment_heatmap,
    plot_workzone_assignment_line,
)
from acbm.assigning.primary_select import select_facility
from acbm.assigning.utils import filter_matrix_to_boundary
from acbm.assigning.work import WorkZoneAssignment
from acbm.logger_config import assigning_primary_locations_logger as logger
from acbm.preprocessing import add_locations_to_activity_chains
from acbm.utils import calculate_rmse

#### LOAD DATA ####

# --- Possible zones for each activity (calculated in 3.1_assign_possible_zones.py)
possible_zones_work = pd.read_pickle(
    acbm.root_path / "data/interim/assigning/possible_zones_work.pkl"
)

# --- boundaries

where_clause = "MSOA21NM LIKE '%Leeds%'"

boundaries = gpd.read_file(
    acbm.root_path / "data/external/boundaries/oa_england.geojson", where=where_clause
)

boundaries = boundaries.to_crs(epsg=4326)

# osm POI data

osm_data_gdf = pd.read_pickle(
    acbm.root_path / "data/interim/assigning/osm_poi_with_zones.pkl"
)
# Convert the DataFrame into a GeoDataFrame, and assign a coordinate reference system (CRS)
osm_data_gdf = gpd.GeoDataFrame(osm_data_gdf, geometry="geometry", crs="EPSG:4326")

# --- Activity chains
activity_chains = pd.read_parquet(
    acbm.root_path / "data/interim/matching/spc_with_nts_trips.parquet"
)
activity_chains = add_locations_to_activity_chains(activity_chains)
activity_chains = activity_chains[activity_chains["TravDay"] == 3]  # Wednesday


activity_chains_work = activity_chains[activity_chains["dact"] == "work"]


# --- WORK: existing travel demand data

# Commuting matrices (from 2021 census)

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


#### ASSIGN TO ZONE FROM FEASIBLE ZONES ####

zone_assignment = WorkZoneAssignment(
    activities_to_assign=possible_zones_work, actual_flows=travel_demand_dict_nomode
)

assignments_df = zone_assignment.select_work_zone_optimization(
    use_percentages=True, weight_max_dev=0.2, weight_total_dev=0.8, max_zones=8
)


# Add assigned zones to activity_chains_work. Replace dzone with assigned_zone
activity_chains_work["dzone"] = activity_chains_work["id"].map(
    assignments_df.set_index("person_id")["assigned_zone"]
)

# --- Evaluating assignment quality

# - RMSE

# Step 1: Convert both the actual demand and the assigned demand data to the correct format
# df: origin_zone, assigned_zone, demand_assigned

# a: Aggregate assignment_opt DataFrame
assignment_agg = (
    assignments_df.groupby(["origin_zone", "assigned_zone"])
    .size()
    .reset_index(name="demand_assigned")
)

# b: Convert travel_demand_dict_no_mode to DataFrame
demand_df = pd.DataFrame(
    list(travel_demand_dict_nomode.items()), columns=["zone_pair", "demand_actual"]
)
demand_df[["origin_zone", "assigned_zone"]] = pd.DataFrame(
    demand_df["zone_pair"].tolist(), index=demand_df.index
)
demand_df.drop(columns=["zone_pair"], inplace=True)


# Step 2: Merge the two DataFrames
workzone_assignment_opt = pd.merge(
    assignment_agg, demand_df, on=["origin_zone", "assigned_zone"], how="outer"
).fillna(0)

# (1) % of Total Demand
workzone_assignment_opt["pct_of_total_demand_actual"] = (
    workzone_assignment_opt["demand_actual"]
    / workzone_assignment_opt["demand_actual"].sum()
) * 100
workzone_assignment_opt["pct_of_total_demand_assigned"] = (
    workzone_assignment_opt["demand_assigned"]
    / workzone_assignment_opt["demand_assigned"].sum()
) * 100

# (2) For each OD pair, demand as % of total demand from the same origin
workzone_assignment_opt["pct_of_o_total_actual"] = workzone_assignment_opt.groupby(
    "origin_zone"
)["demand_actual"].transform(lambda x: (x / x.sum()) * 100)
workzone_assignment_opt["pct_of_o_total_assigned"] = workzone_assignment_opt.groupby(
    "origin_zone"
)["demand_assigned"].transform(lambda x: (x / x.sum()) * 100)

# (3) For each OD pair, demand as % of total demand to each destination
workzone_assignment_opt["pct_of_d_total_actual"] = workzone_assignment_opt.groupby(
    "assigned_zone"
)["demand_actual"].transform(lambda x: (x / x.sum()) * 100)
workzone_assignment_opt["pct_of_d_total_assigned"] = workzone_assignment_opt.groupby(
    "assigned_zone"
)["demand_assigned"].transform(lambda x: (x / x.sum()) * 100)


# Define the output file path
os.makedirs(acbm.root_path / "data/processed/", exist_ok=True)
output_file_path = acbm.root_path / "data/processed/workzone_rmse_results.txt"

# Open the file in write mode
with open(output_file_path, "w") as file:
    # (1) RMSE for % of Total Demand
    predictions = workzone_assignment_opt["pct_of_total_demand_assigned"]
    targets = workzone_assignment_opt["pct_of_total_demand_actual"]

    rmse_pct_of_total_demand = calculate_rmse(predictions, targets)
    file.write(f"RMSE for % of Total Demand: {rmse_pct_of_total_demand}\n")

    # (2) RMSE for demand as % of total demand from the same origin
    predictions = workzone_assignment_opt["pct_of_o_total_assigned"]
    targets = workzone_assignment_opt["pct_of_o_total_actual"]

    rmse_pct_of_o_total = calculate_rmse(predictions, targets)
    file.write(
        f"RMSE for % of Total Demand from the Same Origin: {rmse_pct_of_o_total}\n"
    )

    # (3) RMSE for demand as % of total demand to each destination
    predictions = workzone_assignment_opt["pct_of_d_total_assigned"]
    targets = workzone_assignment_opt["pct_of_d_total_actual"]

    rmse_pct_of_d_total = calculate_rmse(predictions, targets)
    file.write(
        f"RMSE for % of Total Demand to Each Destination: {rmse_pct_of_d_total}\n"
    )

# --- Plots

# Plot the demand_actual and demand_assigned values as a line graph for n origin_zones.
plot_workzone_assignment_line(
    assignment_results=workzone_assignment_opt,
    n=10,
    selection_type="top",
    sort_by="actual",
    save_dir=acbm.root_path / "data/processed/plots/assigning/",
)

# Plot the demand_actual and demand_assigned values as a heatmap for n origin_zones.
plot_workzone_assignment_heatmap(
    workzone_assignment_opt,
    n=20,
    selection_type="top",
    sort_by="assigned",
    save_dir=acbm.root_path / "data/processed/plots/assigning/",
)


#### ASSIGN TO FACILITY ####

# 1. Get neighboring zones

# Sometimes, an activity can be assigned to a zone, but there are no facilities
# in the zone that match the activity type. In this case, we can search for matching
# facilities in neighboring zones.

zone_neighbors = Queen.from_dataframe(boundaries, idVariable="OA21CD").neighbors

# 2. select a facility

# apply the function to a row in activity_chains_ex
activity_chains_work[["activity_id", "activity_geom"]] = activity_chains_work.apply(
    lambda row: select_facility(
        row=row,
        facilities_gdf=osm_data_gdf,
        row_destination_zone_col="dzone",
        row_activity_type_col="dact",
        gdf_facility_zone_col="OA21CD",
        gdf_facility_type_col="activities",
        gdf_sample_col="floor_area",
        neighboring_zones=zone_neighbors,
    ),
    axis=1,
)

# save the activity chains as a pickle

activity_chains_work.to_pickle(
    acbm.root_path / "data/interim/assigning/activity_chains_work.pkl"
)


# --- Plots


# plot the activity chains
plot_desire_lines(
    activities=activity_chains_work,
    activity_type_col="dact",
    activity_type="work",
    bin_size=5000,
    boundaries=boundaries,
    sample_size=1000,
    save_dir=acbm.root_path / "data/processed/plots/assigning/",
)

# plot the scatter plot of actual and reported activities
plot_scatter_actual_reported(
    activities=activity_chains_work,
    activity_type="work",
    x_col="TripTotalTime",
    y_col="length",
    x_label="Reported Travel Time (min)",
    y_label="Actual Distance - Euclidian (km)",
    title_prefix="Scatter plot of TripTotalTime vs. Length",
    save_dir=acbm.root_path / "data/processed/plots/assigning/",
)

plot_scatter_actual_reported(
    activities=activity_chains_work,
    activity_type="work",
    x_col="TripDisIncSW",
    y_col="length",
    x_label="Reported Travel Distance (km)",
    y_label="Actual Distance - Euclidian (km)",
    title_prefix="Scatter plot of TripDisIncSW vs. Length",
    save_dir=acbm.root_path / "data/processed/plots/assigning/",
)
