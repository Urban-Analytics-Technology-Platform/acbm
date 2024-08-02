# # Adding Primary Location to individuals
#
# After assigning an activity chain to each individual, we then need to map these activities to geographic locations. We start with primary locations (work, school) and fill in the gaps later with discretionary locations. This notebook will focus on the primary locations.
#
# We follow the steps outlined in this [github issue](https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/12)

import math
import pickle as pkl

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import LineString

from acbm.assigning import (
    fill_missing_zones,
    get_activities_per_zone,
    get_possible_zones,
    select_activity,
    select_zone,
    zones_to_time_matrix,
)
from acbm.preprocessing import add_location

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

# TODO: Original recoding, no longer required to be applied, consider removing from here
# activity_chains["mode"] = activity_chains["mode"].map(mode_mapping)
# activity_chains["oact"] = activity_chains["oact"].map(purp_mapping)
# activity_chains["dact"] = activity_chains["dact"].map(purp_mapping)


# ### Study area boundaries

boundaries = gpd.read_file("../data/external/boundaries/oa_england.geojson")
boundaries.head(10)


# filter to only include the OA's where "Leeds" is in the MSOA21NM field
boundaries = boundaries[boundaries["MSOA21NM"].str.contains("Leeds", na=False)]


# convert boundaries to 4326
boundaries = boundaries.to_crs(epsg=4326)
# plot the geometry
boundaries.plot()


boundaries.head(10)


# #### Assign activity home locations to boundaries zoning system

# Convert location column in activity_chains to spatial column
# read centroids in source CRS
centroid_layer = pd.read_csv(
    "../data/external/centroids/Output_Areas_Dec_2011_PWC_2022.csv"
)
activity_chains = add_location(
    activity_chains, "EPSG:27700", "EPSG:4326", centroid_layer, "OA11CD", "OA11CD"
)


# Convert the DataFrame into a GeoDataFrame, and assign a coordinate reference system (CRS)
activity_chains = gpd.GeoDataFrame(activity_chains, geometry="location")
activity_chains.crs = "EPSG:4326"  # I assume this is the crs


# plot the boundaries gdf and overlay them with the activity_chains gdf
fig, ax = plt.subplots(figsize=(10, 8))
boundaries.plot(ax=ax, color="lightgrey")
activity_chains.plot(ax=ax, color="red", markersize=1)
plt.title("Activity Chains overlaid on Leeds Output Areas")
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

for _ in range(5):
    print(next(items))


with open("../data/interim/assigning/travel_time_estimates.pkl", "wb") as f:
    pkl.dump(travel_time_estimates, f)


# ### Activity locations
#
# Activity locations are obtained from OSM using the [osmox](https://github.com/arup-group/osmox) package. Check the config documentation in the package and the `config_osmox` file in this repo

# osm data
osm_data = gpd.read_parquet(
    "../data/external/boundaries/west-yorkshire_epsg_4326.parquet"
)


osm_data.head(100)


# get unique values for activties column
osm_data["activities"].unique()


# remove rows with activities = home

osm_data = osm_data[osm_data["activities"] != "home"]
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


# plot the points and then plot the zones on a map
fig, ax = plt.subplots(figsize=(10, 8))
boundaries.plot(ax=ax, color="lightgrey")
osm_data_gdf.plot(ax=ax, color="red", markersize=1)
plt.title("OSM Data overlaid on Leeds Output Areas")
plt.show()


# Let's check if we can use floor area as a weight when sampling a region / a school

# plot the distribution of floor area for rows where activities includes "education_"


# List of activity types
activity_types = [
    "education_kg",
    "education_school",
    "education_university",
    "education_college",
]

# Initialize a list to store DataFrames
df_list = []

# For each activity type, filter the rows where activities includes the activity type, and append to df_list
for activity in activity_types:
    temp_df = osm_data_gdf[
        osm_data_gdf["activities"].apply(lambda x, activity=activity: activity in x)
    ][["floor_area"]].copy()
    temp_df["activity"] = activity
    df_list.append(temp_df)

# Concatenate all the DataFrames in df_list
df = pd.concat(df_list)

# Create a FacetGrid
g = sns.FacetGrid(df, col="activity", col_wrap=2, sharex=False)
g.map(sns.histplot, "floor_area", bins=100)


# To select a zone from a list of zones, we need a list of the activity types that are available in the zone. We then sample probabilistically based on number of activities / total floorspace

# `get_activities_per_zone()` can return a dictionary of dfs, or one big df. Just set `return_df` to `True` to get one df. Let's try both

activities_per_zone_dict = get_activities_per_zone(
    zones=boundaries,
    zone_id_col="OA21CD",
    activity_pts=osm_data,
)


# What does the data look like?

# Get an iterator over the dictionary items
items = iter(activities_per_zone_dict.items())

# Print the first 5 items
for _ in range(5):
    print(next(items))


activities_per_zone = get_activities_per_zone(
    zones=boundaries, zone_id_col="OA21CD", activity_pts=osm_data, return_df=True
)

with open("../data/interim/assigning/activities_per_zone.pkl", "wb") as f:
    pkl.dump(activities_per_zone_dict, f)

# save activities_per_zone as a parquet file
activities_per_zone.to_parquet("../data/interim/assigning/activities_per_zone.parquet")


# ## Education
#
# The NTS gives us the trip duration, mode, and trip purpose of each activity. We have also calculated a zone to zone travel time matrix by mode. We know the locaiton of people's homes so, for home-based activities, we can use this information to determine the feasible zones for each activity.
#
# - Determine activity origin zone, mode, and duration (these are the constraints)
# - Filter travel time matrix to include only destinations that satisfy all constraints. These are the feasible zones
# - If there are no feasible zones, select the zone with the closest travel time to the reported duration
#
# We start with `education` trips as we need to know the trip origin. The vast majority of `education` trips start at home, as shown in `3.1_sandbox-locations_primary.ipynb`. Given that we know the home location of each individual, we can use this information to determine the feasible zones for each education trip.

# ### Getting feasible zones for each activity

print(activity_chains["dact"].value_counts())


activity_chains_edu = activity_chains[activity_chains["dact"] == "education"]


# For education trips, we use age as an indicator for the type of education facility the individual is most likely to go to. The `age_group_mapping` dictionary maps age groups to education facility types. For each person activity, we use the age_group to determine which education facilities to look at.

# map the age_group to an education type (age group is from NTS::Age_B04ID)
# TODO edit osmox config to replace education_college with education_university.
# We should have mutually exclusive groups only and these two options serve the
# same age group
age_group_mapping = {
    1: "education_kg",  # "0-4"
    2: "education_school",  # "5-10"
    3: "education_school",  # "11-16"
    4: "education_university",  # "17-20"
    5: "education_university",  # "21-29"
    6: "education_university",  # "30-39"
    7: "education_university",  # "40-49"
    8: "education_university",  # "50-59"
    9: "education_university",  # "60+"
}


# step 1: age_group mapping onto education type

# map the age_group_mapping dict to an education type (age group is from NTS::Age_B04ID)
activity_chains_edu["education_type"] = activity_chains_edu["age_group"].map(
    age_group_mapping
)
activity_chains_edu.head(3)


possible_zones_school = get_possible_zones(
    activity_chains=activity_chains_edu,
    travel_times=travel_times,
    activities_per_zone=activities_per_zone,
    filter_by_activity=True,
    activity_col="education_type",
    time_tolerance=0.2,
)


# Output is a nested dictionary
for key in list(possible_zones_school.keys())[:10]:
    print(key, " : ", possible_zones_school[key])


# save possible_zones_school to dictionary
with open("../data/interim/assigning/possible_zones_education.pkl", "wb") as f:
    pkl.dump(possible_zones_school, f)


# remove possible_zones_school from environment
# del possible_zones_school

# read in possible_zones_school
possible_zones_school = pd.read_pickle(
    "../data/interim/assigning/possible_zones_education.pkl"
)


# ### Choose a zone for each activity
#
# We choose a zone from the feasible zones. For education trips, we use age as an indicator for the type of education facility the individual is most likely to go to. The `age_group_mapping` dictionary maps age groups to education facility types. For each person activity, we use the age_group to determine which education facilities to look at.
#
# We then sample probabilistically based on the number of facilities in each zone.

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


activity_chains_edu.head(5)


# Total rows and number of rows with NA in dzone
print(f"Total rows: {activity_chains_edu.shape[0]}")
print(
    f"Number of rows with NA in dzone: {activity_chains_edu[activity_chains_edu['dzone'] == 'NA'].shape[0]}"
)


# activity_chains_edu[activity_chains_edu['dzone'] == 'NA']
# what is the mode of the rows with NA in dzone
activity_chains_edu[activity_chains_edu["dzone"] == "NA"]["mode"].value_counts()


# Most of the issue seems to be with walking trips. Let's look further

# Get rows in activity_chains_edu with dzone = NA and mode = walk
filtered_data = activity_chains_edu[
    (activity_chains_edu["dzone"] == "NA") & (activity_chains_edu["mode"] == "walk")
]

# Create bins for TripTotalTime
filtered_data["TripTotalTime_bins"] = pd.cut(
    filtered_data["TripTotalTime"],
    bins=range(0, int(filtered_data["TripTotalTime"].max()) + 5, 5),
)

# Group by TripTotalTime_bins and education_type
grouped_data = filtered_data.groupby(["TripTotalTime_bins", "education_type"]).size()

# Remove groups with zero counts
grouped_data = grouped_data[grouped_data > 0]

# Print the grouped data
print(grouped_data)


# ### Fill in missing zones
#
# Some activities are not assigned a zone because there is no zone that (a) has the activity, and (b) is reachable using the reprted mode and duration (based on travel_time matrix r5 calculations). For these rows, we fill the zone using times based on euclidian distance and estimated speeds
#
#

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


# Total rows and number of rows with NA in dzone
print(f"Total rows: {activity_chains_edu.shape[0]}")
print(
    f"Number of rows with NA in dzone: {activity_chains_edu[activity_chains_edu['dzone'] == 'NA'].shape[0]}"
)


#  ### Assign activity to point locations
#
# After choosing a zone, let's assign the activity to a point location.

activity_chains_ex = activity_chains_edu.copy()


# apply the function to a row in activity_chains_ex
activity_chains_ex[["activity_id", "activity_geom"]] = activity_chains_ex.apply(
    lambda row: select_activity(row, osm_data_gdf, "floor_area"), axis=1
)
activity_chains_ex.head(10)


# #### Plot the results

# For each row in activity_chains_ex, turn the geometry into a linestring: Origin = location and destination = activity_geom

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


# ##### Maps


def plot_activity_chains(
    activities: pd.DataFrame,
    activity_type: str,
    bin_size: int,
    boundaries: gpd.GeoDataFrame,
) -> None:
    """
    Plots activity chains for a given activity type, bin size and geographical boundaries.

    Parameters:
    activities: pd.DataFrame
        A DataFrame containing the activities data. Geometry is a LineString.
    activity_type: str
        The type of activity to plot.
    bin_size: int
        The size of the bins for the histogram. (in meters)
    boundaries: gpd.GeoDataFrame
        A GeoDataFrame containing the geographical boundaries for the plot.

    Returns:
        None
    """
    activities_subset = activities[activities["education_type"] == activity_type]
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

        # Plot the subset
        subset_band.plot(ax=ax, markersize=1)

        # Set the title
        ax.set_title(
            f"{activity_type},\ndistance band: {band},\nNo. of trips: {len(subset_band)} ({percentage:.2f}%)"
        )

    # Remove any unused subplots
    for i in range(len(bands), nrows * ncols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


plot_activity_chains(activity_chains_plot, "education_kg", 5000, boundaries)


plot_activity_chains(activity_chains_plot, "education_school", 5000, boundaries)


plot_activity_chains(activity_chains_plot, "education_university", 5000, boundaries)


# ##### Bar Plots

education_types = activity_chains_plot["education_type"].unique()

# Calculate the number of rows needed for the subplot grid
nrows = int(np.ceil(len(education_types) / 2))

fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 8 * nrows))

# Flatten the axes array to make it easier to iterate over
axs = axs.flatten()

for ax, education_type in zip(axs, education_types):
    subset = activity_chains_plot[
        activity_chains_plot["education_type"] == education_type
    ]
    ax.hist(subset["length"], bins=30, edgecolor="black")
    ax.set_title(f"Activity Chain Lengths for {education_type}")
    ax.set_xlabel("Length")
    ax.set_ylabel("Frequency")

# Remove any unused subplots
for ax in axs[len(education_types) :]:
    ax.remove()

plt.tight_layout()
plt.show()


activity_chains_plot["length"] = activity_chains_plot["length"] / 1000

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Histogram of 'TripDisIncSW'
axs[0].hist(activity_chains_plot["TripDisIncSW"], bins=15, edgecolor="black")
axs[0].set_title("TripDisIncSW (NTS)")

# Histogram of 'length'
axs[1].hist(activity_chains_plot["length"], bins=15, edgecolor="black")
axs[1].set_title("Actual Trip Length (After assigning to location)")

plt.tight_layout()
plt.show()


# ### Logic for assigning people to educational facilities
#
#     for each zone
#         identify individuals with dact = education
#         for each individual
#             get feasible zones (TripTotalTime (NTS) - buffer <= travel time to zone <= TripTotalTime (NTS) + buffer)       # Do we use travel time or distance?
#             if there are feasible zones
#                 if individual_age <= 11
#                     assign individual to random school in feasible zones where type = primary
#                 else if individual_age <= 16 and individual_age > 11
#                     assign individual to random school in feasible zones where type = secondary or technical
#                 else if individual_age > 16 and individual_age <= 18
#                     assign individual to random school in feasible zones where type = college OR university
#                 else
#                     assign individual to random school in feasible zones where type = college OR university OR technical
#             else
#                 assign individual to zone with shortest travel time
#
# - if I have the total number of people enrolled in secondary, technical, college, and university, I can assign make sure that the number of people matched to each educational facility type matches the actual figures. I would use the total numbers and do sampling without replacement
# - I could assign to zones and then use pam to assign to a random facility
#
#
# "All education-related trips from the household travel survey were first split into several groups depending first on the residence area type (see subsubsection 5.1.2) the agent lives in, secondly, on the agent's gender, and, thirdly, on the age of the individual sample who made the trip (and thus on the category of education facility the individual visited: pre-school or elementary school for children aged 14 or less, high school or technical school for teenagers aged 14 to 18, university for people aged 18 to 30 and various places for agents aged 30 or more. For each of these groups, it was then possible to construct the histogram of the distances separating the education place to the home of the individual samples. Finally, a probability density function corresponding to each histogram was obtained." - A synthetic population for the greater São Paulo metropolitan region (Sallard et al 2020)
#
