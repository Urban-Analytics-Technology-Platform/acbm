import geopandas as gpd

import acbm
from acbm.logger_config import preprocessing_logger_logger as logger
from acbm.preprocessing import edit_boundary_resolution, filter_boundaries

# ----- BOUNDARIES
logger.info("Preprocessing Boundary Layer")

## Read in the boundary layer for the whole of England

logger.info("1. Reading in the boundary layer for the whole of England")


boundaries = gpd.read_file(
    acbm.root_path / "data/external/boundaries/oa_england.geojson"
)

boundaries = boundaries.to_crs(epsg=4326)

## Dissolve boundaries if resolution is MSOA

boundary_geography = "OA"  # can only be OA or MSOA
logger.info(f"2. Dissolving boundaries to {boundary_geography} level")

boundaries = edit_boundary_resolution(boundaries, boundary_geography)


## Filter to study area

logger.info("3. Filtering boundaries to specified study area")
# TODO get from config and log
# logger.info(f"3. Filtering boundaries to {config.parameters.boundary_filter_column} = {config.parameters.study_area}")

boundaries_filtered = filter_boundaries(
    boundaries=boundaries, column="LEP22NM1", values=["Leeds City Region"]
)

## Save the output as parquet
logger.info(
    f"4. Saving the boundaries to {acbm.root_path / 'data/external/boundaries/'} path"
)

boundaries_filtered.to_file(
    acbm.root_path / "data/external/boundaries/study_area_zones.geojson",
    driver="GeoJSON",
)
