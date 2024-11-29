from datetime import timedelta

import geopandas as gpd
import pandas as pd
from pam import write
from pam.read import load_travel_diary
from pam.samplers.time import apply_jitter_to_plan
from shapely import Point, wkt

import acbm
from acbm.cli import acbm_cli
from acbm.config import load_config
from acbm.logger_config import converting_to_matsim_logger as logger
from acbm.postprocessing.matsim import (
    add_home_location_to_individuals,
    calculate_percentage_remaining,
    filter_by_pid,
    filter_no_location,
    log_row_count,
)


@acbm_cli
def main(config_file):
    config = load_config(config_file)
    config.init_rng()

    # ----- Read the data

    logger.info("1 - Loading data")

    individuals = pd.read_csv(
        acbm.root_path / "data/processed/activities_pam/people.csv"
    )
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

    # rename age_years to age in individuals
    individuals.rename(columns={"age_years": "age"}, inplace=True)

    # We will be removing some rows in each planning operation. This function helps keep a
    # record of the number of rows in each table after each operation.

    row_counts = []

    logger.info("2.1 - Record number of rows in each df before cleaning")

    log_row_count(individuals, "individuals", "0_initial", row_counts)
    log_row_count(households, "households", "0_initial", row_counts)
    log_row_count(activities, "activities", "0_initial", row_counts)
    log_row_count(legs, "legs", "0_initial", row_counts)
    log_row_count(legs_geo, "legs_geo", "0_initial", row_counts)

    logger.info("2.2 - Remove people that don't exist across all dfs")

    # When writing to matsim using pam, we get an error when a pid exists in one dataset
    #  but not in the other. We will remove these people from the datasets.

    individuals, activities, legs, legs_geo, households = filter_by_pid(
        individuals, activities, legs, legs_geo, households
    )

    log_row_count(individuals, "individuals", "1_filter_by_pid", row_counts)
    log_row_count(households, "households", "1_filter_by_pid", row_counts)
    log_row_count(activities, "activities", "1_filter_by_pid", row_counts)
    log_row_count(legs, "legs", "1_filter_by_pid", row_counts)
    log_row_count(legs_geo, "legs_geo", "1_filter_by_pid", row_counts)

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

    individuals, households, activities, legs, legs_geo = filter_no_location(
        individuals, households, activities, legs, legs_geo
    )

    log_row_count(individuals, "individuals", "2_filter_no_location", row_counts)
    log_row_count(households, "households", "2_filter_no_location", row_counts)
    log_row_count(activities, "activities", "2_filter_no_location", row_counts)
    log_row_count(legs, "legs", "2_filter_no_location", row_counts)
    log_row_count(legs_geo, "legs_geo", "2_filter_no_location", row_counts)

    logger.info("2.5 - Log number of rows in each df after cleaning")

    percentages = calculate_percentage_remaining(row_counts=row_counts)

    # Log the percentages
    for stage, df_name, count, percentage in percentages:
        logger.info(
            f"{df_name} - {stage} - {count} rows: {percentage:.1f}% rows remaining"
        )

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

    # Apply
    individuals_geo = add_home_location_to_individuals(legs_geo, individuals)

    logger.info("4 - Write to MATSim XML")

    logger.info("4.1 - Load travel diary to PAM")

    population = load_travel_diary(
        trips=legs_geo,
        persons_attributes=individuals_geo,
        tour_based=False,
        include_loc=True,
        sort_by_seq=True,
    )

    logger.info("4.2 - Jittering plans")
    # TODO: Move this upstream
    for _, _, person in population.people():
        apply_jitter_to_plan(
            person.plan,
            jitter=timedelta(minutes=config.postprocessing.pam_jitter),
            min_duration=timedelta(minutes=config.postprocessing.pam_min_duration),
        )
        # crop to 24-hours
        person.plan.crop()

    logger.info("4.3 - Write to MATSim XML")

    write.write_matsim_population_v6(
        population=population,
        path=acbm.root_path / "data/processed/activities_pam/plans.xml",
        coordinate_reference_system=f"EPSG:{config.output_crs}",
    )


if __name__ == "__main__":
    main()
