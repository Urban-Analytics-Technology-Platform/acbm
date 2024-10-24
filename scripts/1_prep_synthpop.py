from uatk_spc.builder import Builder

import acbm
from acbm.cli import acbm_cli
from acbm.config import load_config


@acbm_cli
def main(config_file):
    config = load_config(config_file)
    config.init_rng()
    region = config.region

    # Pick a region with SPC output saved
    path = acbm.root_path / "data/external/spc_output/raw/"

    # Add people and households
    spc_people_hh = (
        Builder(path, region, backend="pandas", input_type="parquet")
        .add_households()
        .unnest(
            ["health", "employment", "details", "demographics"], rsuffix="_household"
        )
        .build()
    )
    spc_people_hh.to_parquet(
        acbm.root_path / f"data/external/spc_output/{region}_people_hh.parquet"
    )


if __name__ == "__main__":
    main()
