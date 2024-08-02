#!/usr/bin/env python
# coding: utf-8

# # Adding Work Location to individuals
#
# Assigning individuals to work locations
#
# We follow the steps outlined in this [github issue](https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/12)

import logging
import pickle as pkl
import random

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from libpysal.weights import Queen
from shapely.geometry import Point

from acbm.assigning.assigning import (
    fill_missing_zones,
    filter_matrix_to_boundary,
    get_activities_per_zone,
    get_possible_zones,
    intrazone_time,
    replace_intrazonal_travel_time,
    select_activity,
    select_zone,
    zones_to_time_matrix,
)
from acbm.assigning.work import WorkZoneAssignment

# to display aall columns
pd.set_option("display.max_columns", None)


# ## Load in the data
#

# ### Activity chains

# read parquet file
activity_chains = pd.read_parquet("../data/interim/matching/spc_with_nts_trips.parquet")
activity_chains.head(10)


# #### Data preparation: Mapping trip purposes
#
# Rename columns and map actual modes and trip purposes to the trip table.
#
# Code taken from: https://github.com/arup-group/pam/blob/main/examples/07_travel_survey_to_matsim.ipynb

activity_chains = activity_chains.rename(
    columns={  # rename data
        "JourSeq": "seq",
        "TripOrigGOR_B02ID": "ozone",
        "TripDestGOR_B02ID": "dzone",
        "TripPurpFrom_B01ID": "oact",
        "TripPurpTo_B01ID": "dact",
        "MainMode_B04ID": "mode",
        "TripStart": "tst",
        "TripEnd": "tet",
    }
)


# Check the NTS glossary [here](https://www.gov.uk/government/statistics/national-travel-survey-2022-technical-report/national-travel-survey-2022-technical-report-glossary) to understand what the trip purposes mean.

# add an escort column

mode_mapping = {
    1: "walk",
    2: "cycle",
    3: "car",  #'Car/van driver'
    4: "car",  #'Car/van driver'
    5: "car",  #'Motorcycle',
    6: "car",  #'Other private transport',
    7: "pt",  # Bus in London',
    8: "pt",  #'Other local bus',
    9: "pt",  #'Non-local bus',
    10: "pt",  #'London Underground',
    11: "pt",  #'Surface Rail',
    12: "car",  #'Taxi/minicab',
    13: "pt",  #'Other public transport',
    -10: "DEAD",
    -8: "NA",
}

purp_mapping = {
    1: "work",
    2: "work",  #'In course of work',
    3: "education",
    4: "shop_food",  #'Food shopping',
    5: "shop_other",  #'Non food shopping',
    6: "medical",  #'Personal business medical',
    7: "other_eat_drink",  #'Personal business eat/drink',
    8: "other",  #'Personal business other',
    9: "other_eat_drink",  #'Eat/drink with friends',
    10: "visit",  #'Visit friends',
    11: "other_social",  #'Other social',
    12: "other",  #'Entertain/ public activity',
    13: "other_sport",  #'Sport: participate',
    14: "home",  #'Holiday: base',
    15: "other",  #'Day trip/just walk',
    16: "other",  #'Other non-escort',
    17: "escort_home",  #'Escort home',
    18: "escort_work",  #'Escort work',
    19: "escort_work",  #'Escort in course of work',
    20: "escort_education",  #'Escort education',
    21: "escort_shopping",  #'Escort shopping/personal business',
    22: "escort",  #'Other escort',
    23: "home",  #'Home',
    -10: "DEAD",
    -8: "NA",
}


activity_chains["mode"] = activity_chains["mode"].map(mode_mapping)

activity_chains["oact"] = activity_chains["oact"].map(purp_mapping)

activity_chains["dact"] = activity_chains["dact"].map(purp_mapping)


# ### Study area boundaries
#
# Read in the study area boundaries only (not the whole country)

# SQL-like clause for filtering
where_clause = "MSOA21NM LIKE '%Leeds%'"

boundaries = gpd.read_file(
    "../data/external/boundaries/oa_england.geojson", where=where_clause
)

boundaries.head(10)


# convert boundaries to 4326
boundaries = boundaries.to_crs(epsg=4326)
# plot the geometry
boundaries.plot()


# #### Assign activity home locations to boundaries zoning system

# Convert location column in activity_chains to spatial column

# turn column to shapely point
activity_chains["location"] = activity_chains["location"].apply(
    lambda loc: Point(loc["x"], loc["y"])
)

# Convert the DataFrame into a GeoDataFrame, and assign a coordinate reference system (CRS)
activity_chains = gpd.GeoDataFrame(activity_chains, geometry="location")
activity_chains.crs = "EPSG:4326"  # I assume this is the crs


# plot the boundaries gdf and overlay them with the activity_chains gdf
fig, ax = plt.subplots(figsize=(10, 8))
boundaries.plot(ax=ax, color="lightgrey")
activity_chains.plot(ax=ax, color="red", markersize=1)
plt.title("Home locations overlaid on Leeds Output Areas")
plt.show()


# remove index_right column from activity_chains if it exists
if "index_right" in activity_chains.columns:
    activity_chains = activity_chains.drop(columns="index_right")


# Spatial join to identify which polygons each point is in
activity_chains = gpd.sjoin(
    activity_chains, boundaries[["OA21CD", "geometry"]], how="left", predicate="within"
)
activity_chains = activity_chains.drop("index_right", axis=1)


# ### Travel time matrix for study area
#
# Travel time data between geographical areas (LSOA, OA, custom hexagons etc) is used to determine feasible work / school locations for each individual. The travel times are compared to the travel times of the individual's actual trips from the nts (`tst`/`TripStart` and `tet`/`TripEnd`)

travel_times = pd.read_parquet(
    "../data/external/travel_times/oa/travel_time_matrix_acbm.parquet"
)
travel_times.head(10)


travel_times["combination"].unique()


# Add area code to travel time data

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

travel_times.head(10)


# #### Travel distance matrix
#
# Some areas aren't reachable by specific modes. This can cause problems later on in get_possible_zones() as we won't be able to assign some activities to zones. We create a travel distance matrix to fall back on when there are no travel time calculations

travel_time_estimates = zones_to_time_matrix(
    zones=boundaries, id_col="OA21CD", to_dict=True
)


# What does the data look like?

# Get an iterator over the dictionary items and then print the first n items
items = iter(travel_time_estimates.items())

for i in range(5):
    print(next(items))


with open("../data/interim/assigning/travel_time_estimates.pkl", "wb") as f:
    pkl.dump(travel_time_estimates, f)


# #### Intrazonal trip times
#
# Intrazonal trips all have time = 0. Our `get_possible_zones` function finds zones that are within a specified % threshold from the reported time in the NTS. A threshold percentage from a non zero number never equals 0, so intrazonal trips are not found. The problem is also explained in this [github issue](https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/30)
#
# Below, we assign intrazonal trips a non-zero time based on the zone area

# get intrazone travel time estimates per mode

intrazone_times = intrazone_time(boundaries)

# print first 10 items in the dictionary
items = iter(intrazone_times.items())

for i in range(10):
    print(next(items))


# replace intrazonal travel times with estimates from intrazone_times

travel_times = replace_intrazonal_travel_time(
    travel_times=travel_times,
    intrazonal_estimates=intrazone_times,
    column_to_replace="travel_time_p50",
)


# ### Activity locations
#
# Activity locations are obtained from OSM using the [osmox](https://github.com/arup-group/osmox) package. Check the config documentation in the package and the `config_osmox` file in this repo

# osm data
osm_data = gpd.read_parquet(
    "../data/external/boundaries/west-yorkshire_epsg_4326.parquet"
)


osm_data.head(10)


# get unique values for activties column
osm_data["activities"].unique()


# remove rows with activities = home OR transit

osm_data = osm_data[~osm_data["activities"].isin(["home", "transit"])]
# osm_data = osm_data[osm_data['activities'] != 'home']
osm_data.head(10)


osm_data.activities.unique()


# #### Get the number of activities in each zone
#
# Each zone has a different number of education facilities. We can use the number of facilities in each zone to determine the probability of each zone being chosen for each trip. We can then use these probabilities to randomly assign a zone to each trip.
#
# The education facilities are disaggregated by type. For each activity, we use the individual's age to detemrine which of the following they are most likely to go to
#
# - "kindergarden": education_kg"
# - "school": "education_school"
# - "university": "education_university"
# - "college": "education_college"

# spatial join to identify which zone each point in osm_data is in
osm_data_gdf = gpd.sjoin(
    osm_data, boundaries[["OA21CD", "geometry"]], how="inner", predicate="within"
)
osm_data_gdf.head(5)


boundaries.OA21CD.nunique(), osm_data_gdf.OA21CD.nunique()


# plot the points and then plot the zones on a map
fig, ax = plt.subplots(figsize=(10, 8))
boundaries.plot(ax=ax, color="lightgrey", edgecolor="grey")
osm_data_gdf.plot(ax=ax, column="OA21CD", markersize=1)
plt.title("OSM Data overlaid on Leeds Output Areas")
plt.show()


activities_per_zone = get_activities_per_zone(
    zones=boundaries, zone_id_col="OA21CD", activity_pts=osm_data, return_df=True
)

activities_per_zone


# ### Commuting matrices (from 2021 census)

commute_level = "OA"  # "OA" or "MSOA" data


# Clean the data

if commute_level == "MSOA":
    print("Step 1: Reading in the zipped csv file")
    travel_demand = pd.read_csv("../data/external/ODWP15EW_MSOA_v1.zip")

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
    travel_demand = pd.read_csv("../data/external/ODWP01EW_OA.zip")

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


travel_demand_clipped.head(10)


# Get dictionary of commuting matrices

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


# Get an iterator over the dictionary items
items = iter(travel_demand_dict_nomode.items())

# Print the first 5 items
for i in range(5):
    print(next(items))


# ### Business Registry
#
# Removed for now ...

# ## Workplace Assignment
#
# The NTS gives us the trip duration, mode, and trip purpose of each activity. We have also calculated a zone to zone travel time matrix by mode. We know the locaiton of people's homes so, for home-based activities, we can use this information to determine the feasible zones for each activity.
#
# - Determine activity origin zone, mode, and duration (these are the constraints)
# - Filter travel time matrix to include only destinations that satisfy all constraints. These are the feasible zones
# - If there are no feasible zones, select the zone with the closest travel time to the reported duration

# ### Getting feasible zones for each activity

activity_chains_work = activity_chains[activity_chains["dact"] == "work"]
# Let's focus on a specific day of the week
activity_chains_work = activity_chains_work[
    activity_chains_work["TravDay"] == 3
]  # Wednesday


activity_chains_work.head(10)


possible_zones_work = get_possible_zones(
    activity_chains=activity_chains_work,
    travel_times=travel_times,
    activities_per_zone=activities_per_zone,
    filter_by_activity=True,
    activity_col="dact",
    time_tolerance=0.2,
)


# Output is a nested dictionary
for key in list(possible_zones_work.keys())[:5]:
    print(key, " : ", possible_zones_work[key])


# save possible_zones_school to dictionary
with open("../data/interim/assigning/possible_zones_work.pkl", "wb") as f:
    pkl.dump(possible_zones_work, f)


# remove possible_zones_work from environment
# del possible_zones_work

# read in possible_zones_school
possible_zones_work = pd.read_pickle(
    "../data/interim/assigning/possible_zones_work.pkl"
)


# ### Choose a zone for each activity
#
# We choose a zone from the feasible zones. We have two options:
#
# 1) Iteratively loop over individuals and choose a zone from the feasible zones. We make sure that we don't exceed the flows reported in the commuting matrices
#
# 2) Set up the problem as a Maximum Flow Problem (Optimization)

zone_assignment = WorkZoneAssignment(
    activities_to_assign=possible_zones_work, actual_flows=travel_demand_dict_nomode
)


# #### Option 1: Iterative loop assignment with constraints

# Step 8: Perform the assignment
assignments_df = zone_assignment.select_work_zone_iterative(random_assignment=True)


assignments_df

# count number of weighted and random
# assignments_df['Assignment_Type'].value_counts()


assignments_df.shape[0], activity_chains_work.shape[0]


# count number of None values in Assigned_Zone column
assignments_df["assigned_zone"].isnull().sum()


# #### Option 2: Optimization Problem

assignments_df = zone_assignment.select_work_zone_optimization(
    use_percentages=True, weight_max_dev=0.2, weight_total_dev=0.8, max_zones=8
)

print(assignments_df)


assignments_df.shape[0], activity_chains_work.shape[0]


assignments_df


# #### Evaluating assignment quality

# ##### RMSE

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
workzone_assignment_opt


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

workzone_assignment_opt.head(20)


from math import sqrt

from sklearn.metrics import mean_squared_error

# (1) RMSE for % of Total Demand
rmse_pct_of_total_demand = sqrt(
    mean_squared_error(
        workzone_assignment_opt["pct_of_total_demand_actual"],
        workzone_assignment_opt["pct_of_total_demand_assigned"],
    )
)

# (2) RMSE for demand as % of total demand from the same origin
rmse_pct_of_o_total = sqrt(
    mean_squared_error(
        workzone_assignment_opt["pct_of_o_total_actual"],
        workzone_assignment_opt["pct_of_o_total_assigned"],
    )
)

# (3) RMSE for demand as % of total demand to each destination
rmse_pct_of_d_total = sqrt(
    mean_squared_error(
        workzone_assignment_opt["pct_of_d_total_actual"],
        workzone_assignment_opt["pct_of_d_total_assigned"],
    )
)

print(f"RMSE for % of Total Demand: {rmse_pct_of_total_demand}")
print(f"RMSE for % of Total Demand from the Same Origin: {rmse_pct_of_o_total}")
print(f"RMSE for % of Total Demand to Each Destination: {rmse_pct_of_d_total}")


# ##### Line plots for random subset of zones


def plot_workzone_assignment_line(
    assignment_results: pd.DataFrame,
    n: int,
    selection_type: str = "random",
    sort_by: str = "assigned",
):
    """
    Plot the demand_actual and demand_assigned values for n origin_zones in subplots with two plots per row.
    Home zones can be selected randomly or based on the top actual demand.

    Parameters
    ----------
    assignment_results : DataFrame
        DataFrame containing the actual and assigned demand values.
    n : int
        Number of origin_zones to plot.
    selection_type : str
        Method of selecting origin_zones. Options: 'random', 'top'
        'random': Select n origin_zones randomly,
        'top': Select n home zones with the highest actual demand leaving them.
    sort_by : str
        Column to sort the origin_zones by when selecting the top n. Options: 'actual', 'assigned'
        'actual': Sort by the actual demand, 'assigned': Sort by the assigned

    Returns
    -------
    A matplotlib plot.
    """
    nrows = np.ceil(n / 2).astype(int)
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 6 * nrows))

    if n > 2:
        axes = axes.flatten()
    else:
        axes = np.array([axes]).flatten()

    selected_zones = []
    if selection_type == "random":
        selected_zones = assignment_results["origin_zone"].sample(n).values
    elif selection_type == "top":
        # sort
        top_zones = (
            assignment_results.groupby("origin_zone")[f"demand_{sort_by}"]
            .sum()
            .nlargest(n)
            .index
        )
        selected_zones = top_zones.values

    for i, origin_zone in enumerate(selected_zones):
        origin_zone_df = assignment_results[
            assignment_results["origin_zone"] == origin_zone
        ]

        ax = axes[i]
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_total_demand_actual"],
            "b-",
            label="Actual (% of Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_total_demand_assigned"],
            "b--",
            label="Assigned (% of Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_o_total_actual"],
            "r-",
            label="Actual (% of Origin Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_o_total_assigned"],
            "r--",
            label="Assigned (% of Origin Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_d_total_actual"],
            "g-",
            label="Actual (% of Dest Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_d_total_assigned"],
            "g--",
            label="Assigned (% of Dest Total)",
        )
        ax.set_xlabel("Destination Zone")
        ax.set_ylabel("Demand (%)")
        ax.set_title(
            f"Difference in Actual and Assigned Demand for Origin Zone {origin_zone}"
        )
        ax.legend()
        ax.tick_params(axis="x", rotation=60)

    plt.tight_layout()
    plt.show()


plot_workzone_assignment_line(
    workzone_assignment_opt, 6, selection_type="top", sort_by="actual"
)


plot_workzone_assignment_line(
    workzone_assignment_opt, 6, selection_type="random", sort_by="actual"
)


# ##### Heatmaps


def plot_workzone_assignment_heatmap(
    assignment_results: pd.DataFrame,
    n: int,
    selection_type: str = "random",
    sort_by: str = "assigned",
):
    """
    Create three heatmaps side by side showing the aggregated difference between actual and assigned demand percentages
    for the same n origin_zones across all categories (Global, Origin_Sum, Destination_Sum). The origin_zones are
    either randomly selected or the top n zones with the highest actual demand, consistent across all categories.

    Parameters
    ----------
    assignment_results : DataFrame
        DataFrame containing the actual and assigned demand values.
    n : int
        Number of unique origin_zones to include.
    selection_type : str
        Type of selection for origin_zones. Options: 'random', 'top'.
    sort_by : str
        Column to sort the origin_zones by when selecting the top n. Options: 'actual', 'assigned'
        'actual': Sort by the actual demand, 'assigned': Sort by the assigned

    Returns
    -------
    A matplotlib + seaborn plot.

    """
    categories = ["Global", "Origin", "Destination"]
    fig, axes = plt.subplots(1, len(categories), figsize=(18, 6), sharey=True)

    # Select zones based on selection_type
    if selection_type == "random":
        unique_zones = assignment_results["origin_zone"].unique()
        n = min(n, len(unique_zones))
        selected_zones = np.random.choice(unique_zones, size=n, replace=False)
    elif selection_type == "top":
        # Sort
        top_zones_df = (
            assignment_results.sort_values(by=f"demand_{sort_by}", ascending=False)
            .drop_duplicates("origin_zone")
            .head(n)
        )
        selected_zones = top_zones_df["origin_zone"].values

    for i, category in enumerate(categories):
        prefix_map = {
            "Global": "pct_of_total_demand",
            "Origin": "pct_of_o_total",
            "Destination": "pct_of_d_total",
        }
        prefix = prefix_map[category]

        filtered_df = assignment_results[
            assignment_results["origin_zone"].isin(selected_zones)
        ].copy()
        filtered_df["difference"] = (
            filtered_df[f"{prefix}_actual"] - filtered_df[f"{prefix}_assigned"]
        )
        heatmap_data = filtered_df.pivot_table(
            index="assigned_zone",
            columns="origin_zone",
            values="difference",
            aggfunc=np.mean,
        )

        ax = sns.heatmap(
            heatmap_data, cmap="RdBu", ax=axes[i], cbar=i == len(categories) - 1
        )
        axes[i].set_title(f"Demand Difference: % of {category} Total")
        axes[i].set_xlabel("Home Zone")
        if i == 0:
            axes[i].set_ylabel("Assigned Zone")
        else:
            axes[i].set_ylabel("")

        if i == len(categories) - 1:
            # Create a colorbar with a vertical title
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel(
                "Demand Difference: Actual (%) - Assigned (%)",
                rotation=270,
                labelpad=15,
            )

    plt.tight_layout()
    plt.show()


plot_workzone_assignment_heatmap(
    workzone_assignment_opt, n=20, selection_type="top", sort_by="assigned"
)


plot_workzone_assignment_heatmap(workzone_assignment_opt, n=20, selection_type="random")


# ##### Mapping the differences
#
#

# $$q_{j} = (D_{j} - D_{j}^{obs}) / \sum_{i=1}^{k} D_{k} = (D{j} / \sum_{k} D_{k}) - (D{j}^{obs} / \sum_{k} D_{k}^{obs})$$
#
# - $D_{j}$ is the number of trips going to zone $j$ in the synthetic data, and $D_{j}^{obs}$ is the number of trips going to zone $j$ in the observed data.
# - $D_{k}$ is the total number of trips in the synthetic data, and $D_{k}^{obs}$ is the total number of trips in the observed data.
#

total_actual = workzone_assignment_opt["demand_actual"].sum()
total_assigned = workzone_assignment_opt["demand_assigned"].sum()

# Calculate the total demand and total assigned demand for each destination zone
workzone_assignment_opt_agg = (
    workzone_assignment_opt.groupby("assigned_zone")
    .agg(
        total_demand_actual=("demand_actual", "sum"),
        total_demand_assigned=("demand_assigned", "sum"),
    )
    .reset_index()
)

# Calculate qj (total_demand_actual / total_actual)
workzone_assignment_opt_agg["dj_actual_dk"] = (
    workzone_assignment_opt_agg["total_demand_actual"] / total_actual
)
workzone_assignment_opt_agg["dj_assigned_dk"] = (
    workzone_assignment_opt_agg["total_demand_assigned"] / total_assigned
)
workzone_assignment_opt_agg["qj"] = (
    workzone_assignment_opt_agg["dj_actual_dk"]
    - workzone_assignment_opt_agg["dj_assigned_dk"]
) * 100

# workzone_assignment_opt_agg['qj'] = (workzone_assignment_opt_agg['total_demand_actual'] / total_actual) - (workzone_assignment_opt_agg['total_demand_assigned'] / total_assigned)
workzone_assignment_opt_agg


total_actual, total_assigned


# Add boundary layer to plot

# merge boundaries with workzone_assignment_opt
workzone_assignment_opt_agg_gdf = boundaries[["OA21CD", "geometry"]].merge(
    workzone_assignment_opt_agg, left_on="OA21CD", right_on="assigned_zone", how="left"
)

# Ensure the result is a GeoDataFrame
workzone_assignment_opt_agg_gdf = gpd.GeoDataFrame(workzone_assignment_opt_agg_gdf)
workzone_assignment_opt_agg_gdf


# Plot the map
fig, ax = plt.subplots(figsize=(10, 8))
boundaries.plot(ax=ax, color="lightgrey")
workzone_assignment_opt_agg_gdf.plot(column="qj", legend=True, cmap="coolwarm_r", ax=ax)

plt.title("Qj values for Work Zone Assignment")
plt.show()


# #### Add zones to the activity chains
#

# The left_index in activity_chains_work is the person_id in assignments_df

activity_chains_work = activity_chains_work.merge(
    assignments_df, left_index=True, right_on="person_id", how="left"
)

# drop origin_zone column
# activity_chains_work = activity_chains_work.drop(columns='origin_zone')
# drop original dzone column
activity_chains_work = activity_chains_work.drop(columns="dzone")
# rename assigned_zone to dzone
activity_chains_work = activity_chains_work.rename(columns={"assigned_zone": "dzone"})

activity_chains_work


activity_chains_work.head(5)


#  ### Assign activity to point locations
#
# After choosing a zone, let's assign the activity to a point location.

# #### 1. Get neighboring zones
#
# Sometimes, an activity can be assigned to a zone, but there are no facilities in the zone that match the activity type. In this case, we can search for matching facilities in neighboring zones.

# Assuming zones_gdf is your GeoDataFrame containing the zones
zone_neighbors = Queen.from_dataframe(boundaries, idVariable="OA21CD").neighbors

zone_neighbors


# #### 2. Select a facility
#

from typing import Optional


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
        logging.info(f"Destination zone is NA for row {row.name}")
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
        logging.info(
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
        logging.info(
            f"Found {len(facilities_valid)} matching facilities in neighboring zones"
        )

    # If no specific facilities found and a fallback type is provided, attempt to find facilities matching the fallback type
    if facilities_valid.empty and fallback_type:
        logging.info(
            f"No {row[row_activity_type_col]} facilities in zone {destination_zone} or neighboring zones, trying with {fallback_type}"
        )
        # This should consider both the initial zone and neighboring zones if the previous step expanded the search
        facilities_valid = facilities_in_zone[
            facilities_in_zone[gdf_facility_type_col].apply(
                lambda x: fallback_type in x
            )
        ]
        logging.info(
            f"Found {len(facilities_valid)} matching facilities with type: {fallback_type}"
        )

    # If no facilities found after all attempts, log the failure and return NaN
    if facilities_valid.empty:
        logging.info(
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


activity_chains_ex = activity_chains_work.copy()
# apply the function to a row in activity_chains_ex
activity_chains_ex[["activity_id", "activity_geom"]] = activity_chains_ex.apply(
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

activity_chains_ex.tail(5)


# For each row in activity_chains_ex, turn the geometry into a linestring: Origin = location and destination = activity_geom
from shapely.geometry import LineString

activity_chains_plot = activity_chains_ex.copy()
# filter to only include rows where activity_geom is not NA
activity_chains_plot = activity_chains_plot[
    activity_chains_plot["activity_geom"].notna()
]
activity_chains_plot["line_geometry"] = activity_chains_plot.apply(
    lambda row: LineString([row["location"], row["activity_geom"]]), axis=1
)
# Set the geometry column to 'line_geometry'
activity_chains_plot = activity_chains_plot.set_geometry("line_geometry")

# add the original crs
activity_chains_plot.crs = "EPSG:4326"

# convert crs to metric
activity_chains_plot = activity_chains_plot.to_crs(epsg=3857)
# calculate the length of the line_geometry in meters
activity_chains_plot["length"] = activity_chains_plot["line_geometry"].length

activity_chains_plot.head(10)

# convert crs back to 4326
activity_chains_plot = activity_chains_plot.to_crs(epsg=4326)


activity_chains_plot


# ##### Maps

import math

import matplotlib.patches as mpatches


def plot_activity_chains(
    activities: pd.DataFrame,
    activity_type_col: str,
    activity_type: str,
    bin_size: int,
    boundaries: gpd.GeoDataFrame,
    sample_size: Optional[int] = None,
) -> None:
    """
    Plots activity chains for a given activity type, bin size, geographical boundaries, and an optional sample size.

    Parameters:
    activities: pd.DataFrame
        A DataFrame containing the activities data. Geometry is a LineString.
    activity_type_col: str
        The column name containing the activity type.
    activity_type: str
        The type of activity to plot.
    bin_size: int
        The size of the bins for the histogram. (in meters)
    boundaries: gpd.GeoDataFrame
        A GeoDataFrame containing the geographical boundaries for the plot.
    sample_size: int, optional
        The size of the sample to plot. If None, all data is plotted.

    Returns:
        None
    """
    activities_subset = activities[activities[activity_type_col] == activity_type]

    # If a sample size is specified, sample the activities
    if sample_size is not None and sample_size < len(activities_subset):
        activities_subset = activities_subset.sample(n=sample_size)

    # Mode legend
    modes = activities_subset["mode"].unique()  # Collect all unique modes
    colormap = plt.colormaps.get_cmap("Dark2")  # Generate a colormap
    mode_colors = {
        mode: colormap(i) for i, mode in enumerate(modes)
    }  # Map modes to colors
    legend_patches = [
        mpatches.Patch(color=mode_colors[mode], label=mode) for mode in modes
    ]  # Create legend handles

    # Calculate the number of bins based on the maximum value of 'length'
    num_bins = math.ceil(activities_subset["length"].max() / bin_size)
    # Calculate the bin edges
    bins = np.arange(num_bins + 1) * bin_size
    # Create a new column 'length_band' by cutting 'length' into distance bands
    activities_subset["length_band"] = pd.cut(
        activities_subset["length"], bins, include_lowest=True
    )
    # Get unique bands and sort them
    bands = activities_subset["length_band"].unique()
    bands = sorted(bands, key=lambda x: x.left)
    # Calculate the total number of trips
    total_trips = len(activities_subset)

    # Calculate the number of rows and columns for the subplots
    nrows = math.ceil(len(bands) / 3)
    ncols = 3
    # Create a grid of subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    # Flatten axs for easy iteration
    axs = axs.flatten()

    for ax, band in zip(axs, bands):
        # Get the subset for this band
        subset_band = activities_subset[activities_subset["length_band"] == band]

        # Calculate the percentage of trips in this band
        percentage = len(subset_band) / total_trips * 100

        # Plot the boundaries
        boundaries.plot(ax=ax, color="lightgrey")

        # Plot the subset with correct colors
        for mode in modes:
            # check if mode is in subset_band, and plot if it is
            if mode in subset_band["mode"].unique():
                subset_mode = subset_band[subset_band["mode"] == mode]
                subset_mode.plot(ax=ax, color=mode_colors[mode], label=mode)

        # Set the title
        ax.set_title(
            f"{activity_type},\ndistance band: {band},\nNo. of trips: {len(subset_band)} ({percentage:.2f}%)"
        )

    # Remove any unused subplots
    for i in range(len(bands), nrows * ncols):
        fig.delaxes(axs[i])

    # Place the legend at the bottom
    plt.figlegend(
        handles=legend_patches, loc="lower center", ncol=5, title="Transportation Modes"
    )  # Adjust 'ncol' as needed

    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.03, wspace=0.1, hspace=0.2
    )  # Adjust space to show the legend properly


plot_activity_chains(
    activities=activity_chains_plot,
    activity_type_col="dact",
    activity_type="work",
    bin_size=3000,
    boundaries=boundaries,
    sample_size=1000,
)


activity_chains_plot_walk = activity_chains_plot[activity_chains_plot["mode"] == "walk"]

plot_activity_chains(
    activities=activity_chains_plot_walk,
    activity_type_col="dact",
    activity_type="work",
    bin_size=500,
    boundaries=boundaries,
    sample_size=1000,
)


# ##### Bar Plots

import matplotlib.pyplot as plt

# Calculate the number of rows and columns for the subplots. It is a function of the number of modes
nrows = math.ceil(len(activity_chains_plot["mode"].unique()) / 2)
ncols = 2

# Create a grid of subplots
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))

# Flatten axs for easy iteration
axs = axs.flatten()

# Create a scatter plot for each mode
for i, mode in enumerate(activity_chains_plot["mode"].unique()):
    # Get the subset for this mode
    subset_mode = activity_chains_plot[activity_chains_plot["mode"] == mode]

    # Plot the scatter plot
    ax = axs[i]
    ax.scatter(
        subset_mode["TripTotalTime"], subset_mode["length"] / 1000, alpha=0.3
    )  # Use a single color for all plots

    # Calculate and plot the trend line
    z = np.polyfit(subset_mode["TripTotalTime"], subset_mode["length"] / 1000, 1)
    p = np.poly1d(z)
    ax.plot(subset_mode["TripTotalTime"], p(subset_mode["TripTotalTime"]), "r--")

    ax.set_title(f"Scatter plot of TripTotalTime vs. Length for mode: {mode}")
    ax.set_xlabel("Reported Travel Time (min)")  # Adjusted to km for clarity
    ax.set_ylabel("Actual Distance - Euclidian (km)")


import matplotlib.pyplot as plt

# Calculate the number of rows and columns for the subplots. It is a function of the number of modes
nrows = math.ceil(len(activity_chains_plot["mode"].unique()) / 2)
ncols = 2

# Create a grid of subplots
fig, axs = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))

# Flatten axs for easy iteration
axs = axs.flatten()

# Create a scatter plot for each mode
for i, mode in enumerate(activity_chains_plot["mode"].unique()):
    # Get the subset for this mode
    subset_mode = activity_chains_plot[activity_chains_plot["mode"] == mode]

    # Plot the scatter plot
    ax = axs[i]
    ax.scatter(
        subset_mode["TripDisIncSW"], subset_mode["length"] / 1000, alpha=0.3
    )  # Use a single color for all plots

    # Calculate and plot the trend line
    z = np.polyfit(subset_mode["TripDisIncSW"], subset_mode["length"] / 1000, 1)
    p = np.poly1d(z)
    ax.plot(subset_mode["TripDisIncSW"], p(subset_mode["TripDisIncSW"]), "r--")

    ax.set_title(f"Scatter plot of TripDisIncSW vs. Length for mode: {mode}")
    ax.set_xlabel("Reported Travel Distance (km)")  # Adjusted to km for clarity
    ax.set_ylabel("Actual Distance - Euclidian (km)")


# arrange the activities by the highest difference between the reported travel time and the actual distance
activity_chains_plot["diff"] = activity_chains_plot["TripDisIncSW"] - (
    activity_chains_plot["length"] / 1000
)
activity_chains_plot = activity_chains_plot.sort_values(by="diff", ascending=False)
activity_chains_plot.head(100)


# plot the diff column
plt.figure(figsize=(10, 8))
plt.hist(activity_chains_plot["diff"], bins=200)
plt.xlabel("Difference between Reported Travel Distance and Actual Distance (km)")
plt.ylabel("Frequency")
plt.title(
    "Histogram of the Difference between Reported Travel Distance and Actual Distance"
)
plt.show()
