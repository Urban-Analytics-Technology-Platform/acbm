import os
import re

import click
import numpy as np
import pandas as pd
import polars as pl

from acbm.config import load_and_setup_config


def print_stats(counts, lcol, rcol, stdout=False):
    r2 = np.corrcoef(counts[lcol], counts[rcol])[0, 1] ** 2
    rmse = np.sqrt((counts[lcol] - counts[rcol]).pow(2).mean())
    mes = (counts[lcol] - counts[rcol]).abs().mean()
    if stdout:
        print("Census (total):", counts[lcol].sum())
        print(
            "ACBM (total):",
            counts[rcol].sum(),
        )
        print("The R^2 value is: ", r2)
        print("The RMSE value is: ", rmse)
        print("The MAE value is: ", mes)
    return {"census": counts[lcol].sum(), "acbm": counts[rcol].sum(), "r2": r2}


def get_outs(config, include_wfh=False) -> list[float]:
    acbm_matrix = (
        pl.scan_parquet(config.output_path / "legs_with_locations.parquet")
        .filter(pl.col("purp").eq("work"))
        .unique(subset=["pid"])
        .filter(pl.col("ozone").is_not_null() & pl.col("dzone").is_not_null())
        .group_by(["ozone", "dzone"])
        .len()
        .rename({"len": "acbm_count"})
        .collect()
    )

    census_matrix = (
        pl.scan_csv("data/external/cencus/ODWP15EW_MSOA.csv")
        .rename(
            {
                "Middle layer Super Output Areas code": "ozone",
                "MSOA of workplace code": "dzone",
            }
        )
        .filter(
            pl.col("Place of work indicator (4 categories) code").is_in(
                [1, 3] if include_wfh else [3]
            )
        )
        .rename({"Count": "census_count"})
        .group_by(["ozone", "dzone"])
        .sum()
        .select(["ozone", "dzone", "census_count"])
        .collect()
    )

    # For no normalization
    both_counts = acbm_matrix.join(census_matrix, on=["ozone", "dzone"])

    # Total flows (by origin): i.e. the sum over dzones for a given ozone
    acbm_matrix_norm = acbm_matrix.join(
        acbm_matrix.select(["ozone", "acbm_count"])
        .group_by("ozone")
        .sum()
        .rename({"acbm_count": "acbm_total"}),
        on=["ozone"],
    ).with_columns((pl.col("acbm_count") / pl.col("acbm_total")).alias("acbm_norm"))

    census_matrix_norm = census_matrix.join(
        census_matrix.select(["ozone", "census_count"])
        .group_by("ozone")
        .sum()
        .rename({"census_count": "census_total"}),
        on=["ozone"],
    ).with_columns(
        (pl.col("census_count") / pl.col("census_total")).alias("census_norm")
    )
    both_norms = acbm_matrix_norm.join(census_matrix_norm, on=["ozone", "dzone"])

    return print_stats(both_counts, "census_count", "acbm_count"), print_stats(
        both_norms, "census_norm", "acbm_norm"
    )


@click.command()
@click.option(
    "--config-file-stem",
    type=str,
    default=None,
    required=False,
)
@click.option(
    "--include-wfh",
    is_flag=True,
    default=False,
)
def main(config_file_stem: str | None, include_wfh: bool):
    records = []
    paths = sorted(
        f"./config/{f}"
        for f in os.listdir("./config/")
        if re.search(f"{config_file_stem}_[0-9][0-9]\\.toml$", f)
    )
    if len(paths) == 0:
        msg = f"No config files found for the given stem: '{config_file_stem}'"
        raise ValueError(msg)
    for i, config_file in enumerate(
        paths,
        1,
    ):
        config = load_and_setup_config(config_file)
        try:
            params = [
                "parameters|nts_years",
                # "parameters|nts_regions",
                # "parameters|nts_days_of_week",
                "parameters|tolerance_work",
                "parameters|tolerance_edu",
                "parameters|common_household_day",
                "matching|required_columns",
                # "matching|optional_columns",
                "work_assignment|use_percentages",
                "work_assignment|weight_max_dev",
                "work_assignment|weight_total_dev",
                "work_assignment|max_zones",
            ]

            d1, d2 = get_outs(config, include_wfh)
            record = {}
            record["idx"] = f"{i:02d}"
            record["id"] = config.id
            for p in params:
                record[p.split("|")[-1]] = config.flatten()[p]
            record["census_count"] = d1["census"]
            record["acbm_count"] = d1["acbm"]
            record["r2_count"] = d1["r2"]
            # record["census_norm"] = d2["census"]
            # record["acbm_norm"] = d2["acbm"]
            record["r2_norm"] = d2["r2"]
            records.append(record)
        except Exception as _e:
            # print("----")
            # print(i)
            # print(_e)
            continue

    print(
        pd.DataFrame.from_records(records)
        .sort_values("r2_norm", ascending=False)
        .round(3)
        .to_markdown(index=None)
    )


if __name__ == "__main__":
    main()
