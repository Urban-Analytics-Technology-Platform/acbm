from uatk_spc.builder import Builder

from acbm.cli import acbm_cli
from acbm.config import load_config


@acbm_cli
def main(config_file):
    config = load_config(config_file)
    config.init_rng()
    config.make_dirs()

    # Add people and households
    spc_people_hh = (
        Builder(
            config.spc_raw_path, config.region, backend="pandas", input_type="parquet"
        )
        .add_households()
        .unnest(
            ["health", "employment", "details", "demographics"], rsuffix="_household"
        )
        .build()
    )
    spc_people_hh.to_parquet(config.spc_combined_filepath)


if __name__ == "__main__":
    main()
