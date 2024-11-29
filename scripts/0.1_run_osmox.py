import subprocess

from pyrosm import get_data

from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("preprocessing", __file__)
    logger.info("Getting OSM data")
    fp = get_data(config.region, directory=config.osmox_path)
    logger.info("Running osmox")
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
            f"epsg:{config.output_crs}",
            "-l",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
