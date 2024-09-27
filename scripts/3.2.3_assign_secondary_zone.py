"""
# Adding secondary / discretionary locations to activity chains

This script is used to assign discretionary activities to locations based on a spac-time prism approach. Primary activities (home, work, education) are already assigned to zones. Secondary activities are assigned to zones that are feasible given reported travel times and modes. We use the open-source python library PAM for discretionary activity assignment

- See here for a walkthrough of the PAM functionality: https://github.com/arup-group/pam/blob/main/examples/17_advanced_discretionary_locations.ipynb
- For more info on the spacetime approach for secondary locaiton assignment, see https://www.tandfonline.com/doi/full/10.1080/23249935.2021.1982068
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from pam import write
from pam.planner.od import ODFactory, ODMatrix
from pam.read import load_travel_diary

import acbm
from acbm.assigning.select_zone_secondary import (
    create_od_matrices,
    shift_and_fill_column,
    update_population_plans,
)
from acbm.assigning.utils import (
    activity_chains_for_assignment,
)
from acbm.cli import acbm_cli
from acbm.config import load_config
from acbm.logger_config import assigning_secondary_zones_logger as logger
from acbm.preprocessing import add_location


@acbm_cli
def main(config_file):
    config = load_config(config_file)
    config.init_rng()
    zone_id = config.zone_id

    # --- Load in the data
    logger.info("Loading: activity chains")

    activity_chains = activity_chains_for_assignment()
    activity_chains = activity_chains[activity_chains["TravDay"] == 3]  # Wednesday

    # --- Add OA21CD to the data

    logger.info("Preprocessing: Adding OA21CD to the data")

    where_clause = "MSOA21NM LIKE '%Leeds%'"

    boundaries = gpd.read_file(
        acbm.root_path / "data/external/boundaries/oa_england.geojson",
        where=where_clause,
    )

    # convert boundaries to 4326
    boundaries = boundaries.to_crs(epsg=4326)

    # --- Assign activity home locations to boundaries zoning system

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
        activity_chains,
        boundaries[[config.zone_id, "geometry"]],
        how="left",
        predicate="within",
    )
    activity_chains = activity_chains.drop("index_right", axis=1)

    # remove location column
    activity_chains = activity_chains.drop(columns="location")

    # ----- Primary locations

    logger.info("Loading: data on primary activities")

    def merge_columns_from_other(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
        return df.merge(
            other[
                [
                    col
                    for col in other.columns
                    if col not in df.columns or col in ["id", "seq"]
                ]
            ],
            on=["id", "seq"],
            how="left",
        )

    activity_chains_edu = merge_columns_from_other(
        pd.read_pickle(
            acbm.root_path / "data/interim/assigning/activity_chains_education.pkl"
        ),
        activity_chains,
    )
    activity_chains_work = merge_columns_from_other(
        pd.read_pickle(
            acbm.root_path / "data/interim/assigning/activity_chains_work.pkl"
        ),
        activity_chains,
    )

    # --- Process the data

    # get all activity chains where dact is home
    activity_chains_home = activity_chains[activity_chains["dact"] == "home"]
    # get all activity chains where dact is not work or education
    activity_chains_other = activity_chains[
        ~activity_chains["dact"].isin(["work", "education", "home"])
    ]

    logger.info(
        "Preprocessing: Replacing ozone and dzone with NA in activity_chains_other"
    )
    # Replace ozone and dzone with Na in activity_chains_other. They are incorrect and will be populated later
    activity_chains_other.loc[:, ["ozone", "dzone"]] = np.nan

    logger.info("Preprocessing: Adding dzone for all home activities")
    # replace dzone column with OA21CD. For all home activities, the destination is home
    activity_chains_home["dzone"] = activity_chains_home[config.zone_id]
    activity_chains_home.head(10)

    logger.info("Preprocessing: Combining all activity chains")
    # merge the three dataframes
    activity_chains_all = pd.concat(
        [
            activity_chains_edu,
            activity_chains_work,
            activity_chains_home,
            activity_chains_other,
        ]
    )

    # sort by houshold_id, individual_id, and sequence
    activity_chains_all = activity_chains_all.sort_values(by=["household", "id", "seq"])

    logger.info("Preprocessing: Adding hzone column")
    # Add hzone column (PAM needs one)
    activity_chains_all["hzone"] = activity_chains_all[config.zone_id]

    # TODO find out why some hzone values are NaN
    logger.info("Preprocessing: Filling NaN values in hzone column")
    # Fill NaN values in the hzone column with the first non-NaN value within the same group
    activity_chains_all["hzone"] = activity_chains_all.groupby("id")["hzone"].transform(
        lambda x: x.fillna(method="ffill").fillna(method="bfill")
    )

    logger.info("Preprocessing: Removing people who do not start their day at home")
    # --- Remove all people who do not start their day at home

    # group by id column, and remove all groups where oact is not home in the first row
    activity_chains_all = activity_chains_all.sort_values(by=["household", "id", "seq"])

    logger.info(
        f'PRE-FILTERING: Number of activities: {activity_chains_all.shape[0]}, number of individuals: {activity_chains_all["id"].nunique()}'
    )
    total_activities = activity_chains_all.shape[0]

    activity_chains_all = activity_chains_all.groupby("id").filter(
        lambda x: x.iloc[0]["oact"] == "home"
    )

    logger.info(
        f'POST-FILTERING: Number of activities: {activity_chains_all.shape[0]}, number of individuals: {activity_chains_all["id"].nunique()}'
    )

    removed_activities = total_activities - activity_chains_all.shape[0]
    percentage_removed = (removed_activities / total_activities) * 100
    logger.info(
        f"Removed {removed_activities} activities, which is {percentage_removed:.2f}% of the total activities"
    )

    # --- Edit modes. We can onlyuse modes that we have travel times for
    logger.info("Preprocessing: Editing modes")
    # replace motorcyle with car
    activity_chains_all["mode"] = activity_chains_all["mode"].replace(
        "motorcycle", "car"
    )

    # --- Populate ozone column for primary activities
    logger.info("Preprocessing: Populating ozone column for primary activities")

    # Our dfs have populated the `dzone` column for rows where `dact` matches: [home, work, education].
    # For each person, we look at rows where the `ozone` is one of [home, work, education], and populate the `ozone` column for the primary activity with the same value.

    # Apply the function
    activity_chains_all = shift_and_fill_column(
        data=activity_chains_all,
        group_col="id",
        source_col="dzone",
        target_col="ozone",
        initial_value_col="hzone",
        oact_col="oact",
        hzone_col="hzone",
    )

    logger.info("Preprocessing: (bug) Removing individuals with missing hzone values")
    # Create a boolean mask for groups where there is at least one row with hzone = NA
    mask = activity_chains_all.groupby("id")["hzone"].transform(
        lambda x: x.isna().any()
    )

    logger.info(
        f'Number of individuals to be removed: {activity_chains_all[mask]["id"].nunique()}'
    )
    logger.info(f"Number of activities to be removed: {mask.sum()}")

    # Use the mask to filter out the rows from the original DataFrame
    activity_chains_all = activity_chains_all[~mask]

    activity_chains_all = activity_chains_all[
        [
            "id",
            "household",
            "nts_ind_id",
            "nts_hh_id",
            "age_years",
            "oact",
            "dact",
            "TripTotalTime",
            "TripDisIncSW",
            "seq",
            "mode",
            "tst",
            "tet",
            "ozone",
            "dzone",
            "hzone",
        ]
    ]

    # --- Prepare data for PAM
    logger.info("Preprocessing: Getting data in PAM format")

    # Individuals

    individuals = activity_chains_all[["id", "household", "age_years"]].drop_duplicates(
        subset=["id"]
    )
    individuals = individuals.rename(columns={"id": "pid", "household": "hid"})

    # Households (not necessary)

    # Trips

    trips = activity_chains_all[
        [
            "id",
            "household",
            "seq",
            "hzone",
            "ozone",
            "dzone",
            "dact",
            "mode",
            "tst",
            "tet",
        ]
    ]

    # --- edit the data

    # rename columns
    trips = trips.rename(columns={"id": "pid", "household": "hid", "dact": "purp"})

    # Drop NA values in tst and tet columns and convert to int
    trips = trips.dropna(subset=["tst", "tet"])
    trips["tst"] = trips["tst"].astype(int)
    trips["tet"] = trips["tet"].astype(int)

    # replace Nan values in ozone and dzone with "na"
    trips["ozone"] = trips["ozone"].apply(lambda x: None if pd.isna(x) else x)
    trips["dzone"] = trips["dzone"].apply(lambda x: None if pd.isna(x) else x)
    trips["hzone"] = trips["hzone"].apply(lambda x: None if pd.isna(x) else x)

    # --- Read the population into PAM

    logger.info("Analysis: Reading population into PAM")

    population = load_travel_diary(
        trips=trips,
        persons_attributes=individuals,
        tour_based=False,
        # hhs_attributes = None,
    )

    # --- Discretionary zone selection using PAM

    # Step 1: Preparing travel time and od_probs matrices
    logger.info(
        "Analysis (matrices): Preparing matrices for PAM discretionary activity selection"
    )

    # --- Load travel time estimates
    logger.info("Analysis (matrices): Step 1 - Loading travel time data")

    travel_times = pd.read_parquet(
        acbm.root_path / "data/external/travel_times/oa/travel_time_matrix.parquet"
    )

    # Edit modes
    logger.info("Analysis (matrices): Step 2 - Editing modes")

    # We have travel times for PT by time of day. In discretionary trips, PAM needs the mode column to match the mode labels in ODFactory (see https://github.com/arup-group/pam/blob/main/examples/17_advanced_discretionary_locations.ipynb). We have two options

    # 1. TODO: Preferred: Before reading the population into PAM, edit the mode column of the trips table to replace pt with pt_wkday_morning, pt_wkday_evening etc depending on day and time of trip. I dont know if this will work downstream
    # 2. Simplify our travel time data. Use the same travel time regardless of time of day, and label as pt (to match with mode column)

    # I will do 2 for now

    # Check if 'time_of_day' column exists (this implies we have travel times for PT by time of day - ie travel times have not
    # been generated by zones_to_time_matrix() function)
    # TODO: just replace with time estimates from zones_to_time_matrix() function
    if "time_of_day" in travel_times.columns:
        # Apply filtering logic
        travel_times = travel_times[
            (travel_times["mode"] != "pt")
            | (
                (travel_times["mode"] == "pt")
                & (travel_times["time_of_day"] == "morning")
                & (travel_times["weekday"] == 1)
            )
        ]

    # Rename specific values in "mode" column
    travel_times["mode"] = travel_times["mode"].replace({"cycle": "bike"})

    # --- Calculate OD probabilities (probabilities of choosing a destination zone for an activity, given the origin zone)
    logger.info("Analysis (matrices): Step 3 - Calculating OD probabilities")

    activities_per_zone = pd.read_parquet(
        acbm.root_path / "data/interim/assigning/activities_per_zone.parquet"
    )

    # keep only rows that don't match primary activities
    activities_per_zone = activities_per_zone[
        activities_per_zone["activity"].isin(["shop", "other", "medical", "visit"])
    ]

    # group by zone and get sum of counts and floor_area
    activities_per_zone = (
        activities_per_zone.groupby(config.zone_id)
        .agg({"counts": "sum", "floor_area": "sum"})
        .reset_index()
    )

    # Merge to get floor_area for origin
    merged_df = travel_times.merge(
        activities_per_zone, left_on="OA21CD_to", right_on=config.zone_id
    )

    # Calculate the visit_probability: it is a funciton of floor_area and travel time
    merged_df["visit_prob"] = np.where(
        merged_df["time"] != 0,  # avoid division by zero
        round(merged_df["floor_area"] / np.sqrt(merged_df["time"])),
        round(merged_df["floor_area"]),
    )

    # --- Create matrices for travel times and OD probabilities
    logger.info(
        "Analysis (matrices): Step 4 - Creating matrices for travel times and OD probabilities"
    )

    # Get unique zone labels for matrix
    # TODO: get these from boundary/zone layer instead
    zone_labels = pd.unique(
        travel_times[
            [
                config.origin_zone_id(zone_id),
                config.destination_zone_id(zone_id),
            ]
        ].values.ravel("K")
    )
    zone_labels = tuple(zone_labels)  # PAM function needs a tuple

    matrix_travel_times = create_od_matrices(
        df=merged_df,
        mode_column="mode",
        value_column="time",
        zone_labels=zone_labels,
        fill_value=300,  # replace missing travel times with 6 hours (they are unreachable)
        zone_from=config.origin_zone_id(zone_id),
        zone_to=config.destination_zone_id(zone_id),
    )

    matrix_od_probs = create_od_matrices(
        df=merged_df,
        mode_column="mode",
        value_column="visit_prob",
        zone_labels=zone_labels,
        # replace missing probabilities with 1. There are no activities so shouldn't be visited
        # 1 used instead of 0 to avoid (ValueError: Total of weights must be finite) in weighted sampling
        # (https://github.com/arup-group/pam/blob/c8bff760fbf92f93f95ff90e4e2af7bbe107c7e3/src/pam/planner/utils_planner.py#L17)
        fill_value=1,
        zone_from=config.origin_zone_id(zone_id),
        zone_to=config.destination_zone_id(zone_id),
    )

    # Create ODMatrix objects
    logger.info("Analysis (matrices): Step 5 - Creating ODMatrix objects")

    mode_types = travel_times["mode"].unique()

    matrices_pam_travel_time = [
        ODMatrix("time", mode, zone_labels, zone_labels, matrix_travel_times[mode])
        for mode in mode_types
    ]

    matrices_pam_od_probs = [
        ODMatrix("od_probs", mode, zone_labels, zone_labels, matrix_od_probs[mode])
        for mode in mode_types
    ]

    # combine ODMatrix objects
    matrices_pam_all = matrices_pam_travel_time + matrices_pam_od_probs

    # create ODFactory
    logger.info("Analysis (matrices): Step 6 - Creating ODFactory object")

    od = ODFactory.from_matrices(matrices=matrices_pam_all)

    # --- Fill in zones for secondary activities
    logger.info("Analysis (assigning): Filling in zones for secondary activities!")

    update_population_plans(population, od)

    # --- Save
    logger.info("Saving: Step 7 - Saving population")

    write.to_csv(population, dir=(acbm.root_path / "data/processed/activities_pam"))


if __name__ == "__main__":
    main()
