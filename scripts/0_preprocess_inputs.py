import geopandas as gpd
import pandas as pd
from uatk_spc import Reader

import acbm
from acbm.cli import acbm_cli
from acbm.config import load_config
from acbm.logger_config import preprocessing_logger as logger
from acbm.preprocessing import edit_boundary_resolution


@acbm_cli
def main(config_file):
    config = load_config(config_file)
    config.init_rng()
    region = config.region
    # Pick a region with SPC output saved
    spc_path = acbm.root_path / "data/external/spc_output/raw/"

    # ----- BOUNDARIES
    logger.info("Preprocessing Boundary Layer")

    ## Read in the boundary layer for the whole of England

    logger.info("1. Reading in the boundary layer for the whole of England")

    boundaries = gpd.read_file(
        acbm.root_path / "data/external/boundaries/oa_england.geojson"
    )

    boundaries = boundaries.to_crs(epsg=f"epsg:{config.output_crs}")

    ## --- Dissolve boundaries if resolution is MSOA

    boundary_geography = config.parameters.boundary_geography  # can only be OA or MSOA
    logger.info(f"2. Dissolving boundaries to {boundary_geography} level")

    boundaries = edit_boundary_resolution(
        study_area=boundaries, geography=boundary_geography, zone_id=config.zone_id
    )

    ## --- Filter to study area
    # we filter using msoa21cd values, which exist regardless of the boundary resolution

    logger.info("3. Filtering boundaries to specified study area")

    # Step 1: Get zones from SPC (these will be 2011 MSOAs)
    spc = Reader(spc_path, region, backend="pandas")
    zones_in_region = list(spc.info_per_msoa.keys())

    # Step 2: Filter boundaries to identified zones

    # a) get MSOA11CD to MSOA21CD lookup
    msoa_lookup = pd.read_csv(
        acbm.root_path
        / "data/external/MSOA_2011_MSOA_2021_Lookup_for_England_and_Wales.csv"
    )
    # Filter msoa_lookup to include only rows where MSOA11CD is in zones_in_region
    msoa_lookup_filtered = msoa_lookup[msoa_lookup["MSOA11CD"].isin(zones_in_region)]
    # Extract the corresponding MSOA21CD values
    msoa21cd_values = msoa_lookup_filtered["MSOA21CD"].tolist()

    # b) filter boundaries to include only rows where MSOA21CD is in msoa21cd_values
    boundaries_filtered = boundaries[boundaries["MSOA21CD"].isin(msoa21cd_values)]

    ## Save the output as parquet
    logger.info(
        f"4. Saving the boundaries to {acbm.root_path / 'data/external/boundaries/'} path"
    )

    boundaries_filtered.to_file(
        acbm.root_path / "data/external/boundaries/study_area_zones.geojson",
        driver="GeoJSON",
    )


if __name__ == "__main__":
    main()
