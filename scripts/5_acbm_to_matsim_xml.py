import geopandas as gpd
import pandas as pd
from pam import write
from pam.read import load_travel_diary
from shapely import Point, wkt

import acbm
from acbm.logger_config import converting_to_matsim_logger as logger

# ----- Read the data

logger.info("1 - Loading data")

individuals = pd.read_csv(acbm.root_path / "data/processed/activities_pam/people.csv")
households = pd.read_csv(
    acbm.root_path / "data/processed/activities_pam/households.csv"
)
activities = pd.read_csv(
    acbm.root_path / "data/processed/activities_pam/activities.csv"
)
legs = pd.read_csv(acbm.root_path / "data/processed/activities_pam/legs.csv")
legs_geo = pd.read_parquet(
    acbm.root_path / "data/processed/activities_pam/legs_with_locations.parquet"
)


# ----- Clean the data

logger.info("2 - Cleaning data")


# We will be removing some rows in each planning operation. This function helps keep a
# record of the number of rows in each table after each operation.

row_counts = []


# Function to log row counts
def log_row_count(df, name, operation):
    row_counts.append((operation, name, len(df)))


logger.info("2.1 - Record number of rows in each df before cleaning")

log_row_count(individuals, "individuals", "0_initial")
log_row_count(households, "households", "0_initial")
log_row_count(activities, "activities", "0_initial")
log_row_count(legs, "legs", "0_initial")
log_row_count(legs_geo, "legs_geo", "0_initial")

logger.info("2.2 - Remove people that don't exist across all dfs")

# When writing to matsim using pam, we get an error when a pid exists in one dataset
#  but not in the other. We will remove these people from the datasets.


def filter_by_pid(individuals, activities, legs, legs_geo, households):
    """
    Filter the input DataFrames to include only include people (pids) that exist in all
    dfs

    Parameters
    ----------
    individuals: pd.DataFrame
        Individuals DataFrame.
    activities: pd.DataFrame
        Activities DataFrame.
    legs: pd.DataFrame:
        Legs DataFrame.
    legs_geo: pd.DataFrame
        Legs with geo DataFrame.
    households: pd.DataFrame
        Households DataFrame.

    Returns
    -------
    tuple
        A tuple containing the filtered DataFrames (individuals, activities, legs, legs_geo, households).
    """
    # Identify common pids
    common_pids = (
        set(individuals["pid"])
        .intersection(activities["pid"])
        .intersection(legs["pid"])
        .intersection(legs_geo["pid"])
    )

    # Filter Individual Level DataFrames
    individuals = individuals[individuals["pid"].isin(common_pids)]
    activities = activities[activities["pid"].isin(common_pids)]
    legs = legs[legs["pid"].isin(common_pids)]
    legs_geo = legs_geo[legs_geo["pid"].isin(common_pids)]

    # Filter Household Level DataFrame
    households = households[households["hid"].isin(individuals["hid"])]

    return individuals, activities, legs, legs_geo, households


# Apply
individuals, activities, legs, legs_geo, households = filter_by_pid(
    individuals, activities, legs, legs_geo, households
)

log_row_count(individuals, "individuals", "1_filter_by_pid")
log_row_count(households, "households", "1_filter_by_pid")
log_row_count(activities, "activities", "1_filter_by_pid")
log_row_count(legs, "legs", "1_filter_by_pid")
log_row_count(legs_geo, "legs_geo", "1_filter_by_pid")


logger.info("2.3 - Rename geometry columns (for PAM)")
# TODO: Rename columns upstream in 3.3_assign_facility_all script
legs_geo.rename(
    columns={
        "start_location_geometry_wkt": "start_loc",
        "end_location_geometry_wkt": "end_loc",
    },
    inplace=True,
)

logger.info("2.4 - Remove people with missing location data ")


def filter_no_location(individuals, households, activities, legs, legs_geo):
    """
    Cleans the provided DataFrames by removing rows without location data. Gets all pids
    that have at least one row with missing location data, and removes all rows with
    these pids. pids are geneerated from two sources:
       - legs_geo with missing start_loc or end_loc
       - individuals with missing hzone

    Parameters
    ----------
    individuals : pd.DataFrame
        DataFrame containing individual data.
    households : pd.DataFrame
        DataFrame containing household data.
    activities : pd.DataFrame
        DataFrame containing activity data.
    legs : pd.DataFrame
        DataFrame containing legs data.
    legs_geo : pd.DataFrame
        DataFrame containing legs with geographic data.

    Returns
    -------
    tuple
        A tuple containing the cleaned DataFrames
        (individuals_cleaned, households_cleaned, activities_cleaned, legs_cleaned, legs_geo_cleaned).
    """
    # Identify rows in legs_geo where start_loc or end_loc are null
    invalid_rows_legs_geo = legs_geo[
        legs_geo["start_loc"].isnull() | legs_geo["end_loc"].isnull()
    ]

    # Extract the pid values associated with these rows
    invalid_pids_legs_geo = invalid_rows_legs_geo["pid"].unique()

    # Identify rows in individuals where hzone is null
    invalid_rows_individuals = individuals[individuals["hzone"].isnull()]

    # Extract the pid values associated with these rows
    invalid_pids_individuals = invalid_rows_individuals["pid"].unique()

    # Combine the invalid pid values from both sources
    invalid_pids = set(invalid_pids_legs_geo).union(set(invalid_pids_individuals))

    # Remove rows with these pids from all DataFrames
    individuals_cleaned = individuals[~individuals["pid"].isin(invalid_pids)]
    activities_cleaned = activities[~activities["pid"].isin(invalid_pids)]
    legs_cleaned = legs[~legs["pid"].isin(invalid_pids)]
    legs_geo_cleaned = legs_geo[~legs_geo["pid"].isin(invalid_pids)]

    # Extract remaining hid values from individuals_cleaned
    remaining_hids = individuals_cleaned["hid"].unique()

    # Filter households_cleaned to only include rows with hid values in remaining_hids
    households_cleaned = households[households["hid"].isin(remaining_hids)]

    return (
        individuals_cleaned,
        households_cleaned,
        activities_cleaned,
        legs_cleaned,
        legs_geo_cleaned,
    )


# Apply
individuals, households, activities, legs, legs_geo = filter_no_location(
    individuals, households, activities, legs, legs_geo
)


log_row_count(individuals, "individuals", "2_filter_no_location")
log_row_count(households, "households", "2_filter_no_location")
log_row_count(activities, "activities", "2_filter_no_location")
log_row_count(legs, "legs", "2_filter_no_location")
log_row_count(legs_geo, "legs_geo", "2_filter_no_location")


logger.info("2.5 - Log number of rows in each df after cleaning")


def calculate_percentage_remaining(row_counts):
    """
    Calculate the percentage of rows remaining for each DataFrame based on the
    initial counts.

    Parameters
    ----------
    row_counts : list of tuples
        List of tuples containing stage, DataFrame names,
        and their row counts.

    Returns
    -------
    list of tuples
        List of tuples containing stage, DataFrame names, and their percentage
        of rows remaining.
    """
    # Extract initial counts
    initial_counts = {
        df_name: count for stage, df_name, count in row_counts if stage == "0_initial"
    }

    # Calculate percentage remaining
    percentage_remaining = []
    for stage, df_name, count in row_counts:
        if df_name in initial_counts:
            initial_count = initial_counts[df_name]
            percentage = round((count / initial_count) * 100, 1)
            percentage_remaining.append((stage, df_name, count, percentage))

    # Sort by df_name
    percentage_remaining.sort(key=lambda x: x[1])

    return percentage_remaining


percentages = calculate_percentage_remaining(row_counts)


# Log the percentages
for stage, df_name, count, percentage in percentages:
    logger.info(f"{df_name} - {stage} - {count} rows: {percentage:.1f}% rows remaining")


logger.info("3a - Convert geometry columns to POINT geometry")


# Function to convert to Point if not already a Point
def convert_to_point(value):
    if isinstance(value, Point):
        return value
    return wkt.loads(value)


# Convert start_loc and end_loc to shapely point objects
legs_geo["start_loc"] = legs_geo["start_loc"].apply(convert_to_point)
legs_geo["end_loc"] = legs_geo["end_loc"].apply(convert_to_point)

# Convert to GeoDataFrame with start_loc as the active geometry
legs_geo = gpd.GeoDataFrame(legs_geo, geometry="start_loc")


logger.info("3b - Add home location to individuals")


def add_home_location_to_individuals(legs_geo, individuals):
    """
    Adds home location to individuals dataframe. Location is obtained
    from legs_geo (rows with orign activity = home)

    Parameters
    ----------
    legs_geo : pd.DataFrame
        DataFrame containing legs with geographic data.
    individuals : pd.DataFrame
        DataFrame containing individual data.

    Returns
    -------
    pd.DataFrame
        The modified individuals DataFrame with location information.
    """
    # Filter by origin activity = home
    legs_geo_home = legs_geo[legs_geo["origin activity"] == "home"]

    # Get one row for each hid group
    legs_geo_home = legs_geo_home.groupby("hid").first().reset_index()

    # Keep only the columns we need: hid and start_location
    legs_geo_home = legs_geo_home[["hid", "start_loc"]]

    # Rename start_loc to loc
    legs_geo_home.rename(columns={"start_loc": "loc"}, inplace=True)

    # Merge legs_geo_home with individuals
    individuals_geo = individuals.copy()
    individuals_geo = individuals_geo.merge(legs_geo_home, on="hid")

    # Remove rows with missing loc
    return individuals_geo[individuals_geo["loc"].notnull()]


# Apply
individuals_geo = add_home_location_to_individuals(legs_geo, individuals)


logger.info("4 - Write to MATSim XML")

logger.info("4.1 - Load travel diary to PAM")

population = load_travel_diary(
    trips=legs_geo,
    persons_attributes=individuals,
    tour_based=False,
    include_loc=True,
    sort_by_seq=True,
    # hhs_attributes = None,
)

logger.info("4.2 - Write to MATSim XML")

write.write_matsim_population_v6(
    population=population,
    path=acbm.root_path / "data/processed/activities_pam/plans.xml",
)
