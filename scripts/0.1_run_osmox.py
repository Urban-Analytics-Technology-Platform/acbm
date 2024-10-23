import os
import subprocess

from pyrosm import get_data

import acbm
from acbm.cli import acbm_cli
from acbm.config import load_config


@acbm_cli
def main(config_file):
    config = load_config(config_file)
    config.init_rng()
    os.makedirs(acbm.root_path / "data/external/osmox", exist_ok=True)
    _fp = get_data(config.region, directory=acbm.root_path / "data/external/osmox")
    subprocess.run(
        [
            "osmox",
            "run",
            acbm.root_path / "osmox/config_osmox.json",
            acbm.root_path / f"data/external/osmox/{config.region}.osm.pbf",
            f"data/external/osmox/{config.region}",
            "-f",
            "geoparquet",
            "-crs",
            "epsg:27700",
            "-l",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
