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

    # People and time-use data
    # Subset of (non-time-use) features to include and unnest
    # The features can be found here: https://github.com/alan-turing-institute/uatk-spc/blob/main/synthpop.proto
    features = {
        "health": [
            "bmi",
            "has_cardiovascular_disease",
            "has_diabetes",
            "has_high_blood_pressure",
            "self_assessed_health",
            "life_satisfaction",
        ],
        "demographics": ["age_years", "ethnicity", "sex", "nssec8"],
        "employment": ["sic1d2007", "sic2d2007", "pwkstat", "salary_yearly"],
    }

    # build the table
    spc_people_tu = (
        Builder(path, region, backend="polars", input_type="parquet")
        .add_households()
        .add_time_use_diaries(features, diary_type="weekday_diaries")
        .build()
    )

    # save the output
    spc_people_tu.write_parquet(
        acbm.root_path / f"data/external/spc_output/{region}_people_tu.parquet"
    )


if __name__ == "__main__":
    main()
