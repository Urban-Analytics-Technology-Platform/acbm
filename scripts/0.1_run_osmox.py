import os
import subprocess

import geopandas as gpd
import requests
from pyrosm import OSM, get_data

from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("preprocessing", __file__)
    logger.info("Getting OSM data")
    try:
        fp = get_data(config.region, directory=config.osmox_path)
    except Exception as e:
        logger.error(e)
        logger.info(f"Trying manual download of region: {config.region}")
        url = f"http://download.geofabrik.de/europe/united-kingdom/england/{config.region}-latest.osm.pbf"
        filename = url.split("/")[-1]
        fp = os.path.join(config.osmox_path, filename)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(fp, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Region ({config.region}) successfully downloaded to: {fp}")
    # Use the CRS value from the config
    crs_value = f"epsg:{config.output_crs}"
    logger.info(f"Running osmox with output crs: {crs_value}")
    subprocess.run(
        [
            "osmox",
            "run",
            # TODO: add to lib as string
            config.root_path / "osmox/config_osmox.json",
            fp,
            config.osmox_path / config.region,
            "-f",
            "geoparquet",
            "-crs",
            # For the distances to be accurate, needs to be same CRS as OSM data for the region.
            # However, distances from osmox are currently not used in the pipeline so any CRS will work.
            # In general, the CRS is transformed in the pipeline when this data is used.
            # See: https://github.com/arup-group/osmox/blob/82602d411374ebc9fd33443f8f7c9816b63715ec/docs/osmox_run.md#L35-L38
            crs_value,
            "-l",
        ],
        check=False,
    )
    logger.info("osmox run complete")

    logger.info("Assigning linkID to facilities")
    # Add linkID column (closest road to the facility). See #https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/109

    logger.info("Step 1: reading the poi data prepared by osmox")
    # Read in the poi data
    poi_fp = os.path.join(
        config.osmox_path, f"{config.region}_epsg_{config.output_crs}.parquet"
    )
    pois = gpd.read_parquet(poi_fp)

    logger.info("Step 2: reading the osm road network data from the pbf file")
    # Read in the road data
    osm = OSM(fp)
    osm_roads = osm.get_network(network_type="driving")
    osm_roads = osm_roads.to_crs(crs_value)

    logger.info(
        "Step 3: Addign a road linkID to each facility (based on the closest osm link)"
    )
    # Find the nearest road to each facility
    pois_with_links = gpd.sjoin_nearest(
        pois,
        osm_roads[
            ["id", "geometry"]
        ],  # Only include the 'id' and 'geometry' columns from osm_roads
        how="left",
        max_distance=1000,
        lsuffix="",  # No suffix for pois_sample
        # rsuffix="roads"   # No suffix for osm_roads
    ).drop(columns=["index_right"])

    # Rename the columns to remove the trailing _ created by the sjoin
    pois_with_links = pois_with_links.rename(
        columns={"id_": "id", "id_right": "linkId"}
    )

    pois_with_links = pois_with_links.reset_index(drop=True)

    logger.info(
        "Step 4: Saving the POIs with links, overwriting the original pois file"
    )
    # Save the file (overwrite the original pois file)
    pois_with_links.to_parquet(poi_fp, index=False)

    logger.info(f"pois_with_links saved to: {poi_fp}")


if __name__ == "__main__":
    main()
