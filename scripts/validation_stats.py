import glob
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import polars as pl

import acbm
from acbm.config import Config, load_and_setup_config, load_config


def config_summary():
    configs = {}
    for path in glob.glob("data/outputs/*"):
        try:
            config = load_and_setup_config(Path(path) / "config.toml")
            configs[config.id] = config.flatten()
        except Exception:
            continue
    print(pd.DataFrame.from_records(configs).to_markdown())


def print_stats(counts, lcol, rcol):
    print("Census (total):", counts[lcol].sum())
    print(
        "ACBM (total):",
        counts[rcol].sum(),
    )
    r2 = np.corrcoef(counts[lcol], counts[rcol])[0, 1] ** 2
    print("The R^2 value is: ", r2)
    rmse = np.sqrt((counts[lcol] - counts[rcol]).pow(2).mean())
    print("The RMSE value is: ", rmse)
    mes = (counts[lcol] - counts[rcol]).abs().mean()
    print("The MAE value is: ", mes)


def print_stats_work(config: Config):
    # SPC
    print()
    print("Expected compared to actual numbers of people travelling to work...")
    print(
        "Total expected people travelling to work (SPC pwkstat 1 or 2): {:,.0f}".format(
            pl.scan_parquet(config.spc_combined_filepath)
            .select(pl.col("pwkstat").is_in([1, 2]).sum())
            .collect()
            .to_numpy()
            .squeeze()
        )
    )

    # SPC with nts trips (after matching)
    print("\nSPC with NTS trips (subsetted to chosen day)")
    from acbm.assigning.utils import activity_chains_for_assignment

    acs_no_subset = pl.DataFrame(
        activity_chains_for_assignment(config, subset_to_chosen_day=False)
    )
    acs = pl.DataFrame(
        activity_chains_for_assignment(config, subset_to_chosen_day=True)
    )
    print(
        "Total with travel day: {:,.0f} ({:.1%}), total: {:,.0f}".format(
            acs.unique("id").shape[0],
            acs.unique("id").shape[0] / acs_no_subset.unique("id").shape[0],
            acs_no_subset.unique("id").shape[0],
        )
    )
    df = (
        acs.select(["id", "dact"])
        .group_by("id")
        .agg(pl.lit("work").is_in(pl.col("dact")).alias("any_work_activity"))
    )
    any_work_activity = df.select("any_work_activity").sum().to_numpy().squeeze()
    print(
        "Any work activity: {:,.0f} ({:,.1%}), Total people: {:,.0f}".format(
            any_work_activity,
            (any_work_activity / acs.unique("id").shape[0]),
            acs.unique("id").shape[0],
        )
    )

    # Final output (legs.csv)
    print("\nFinal legs.csv")
    legs = pl.scan_csv(config.output_path / "legs.csv")
    df = (
        legs.select(["pid", "purp"])
        .group_by("pid")
        .agg(pl.lit("work").is_in(pl.col("purp")).alias("any_work_activity"))
        .collect()
    )
    any_work_activity = df.select("any_work_activity").sum().to_numpy().squeeze()
    print(
        "Any work activity: {:,.0f} ({:,.1%}), Total people (in legs.csv): {:,.0f}".format(
            any_work_activity,
            (any_work_activity / df.unique("pid").shape[0]),
            df.unique("pid").shape[0],
        )
    )


@click.command()
@click.option(
    "--id",
    # prompt="Run ID for stats to be generated from",
    type=str,
    required=False,
    default=None,
)
@click.option(
    "--config-file",
    # prompt="Config file to generate run ID for stats to be generated from",
    type=str,
    default=None,
    required=False,
)
# To view the output table, pipe into less with -S
# $ python scripts/validation_stats.py --summary | less -S
@click.option(
    "--summary",
    "-s",
    is_flag=True,
    required=False,
)
@click.option("--scaling", type=float, default=1.0, required=False)
def main(id: str | None, config_file: str | None, summary: bool, scaling: float):
    if summary:
        config_summary()
        return

    if id is not None and config_file is not None:
        print("Specify one of 'id' and 'config-file'.")
        exit(1)
    pd.options.mode.copy_on_write = True
    os.chdir(acbm.root_path)
    if config_file is None:
        config = load_config(f"data/outputs/{id}/config.toml")
    else:
        config = load_config(config_file)
    trav_day = 3
    spc = pl.read_parquet(
        config.interim_path / f"{config.region.lower()}_people_hh.parquet"
    )
    df = pl.read_parquet(
        config.interim_path / "matching" / "spc_with_nts_trips.parquet"
    ).join(spc.select(["id", "pwkstat"]), on="id", how="left", coalesce=True)

    print(f"> Config ID: {config.id}")
    print("Summary stats for SPC and NTS matched people and activities...")
    print(
        "% of people with a NTS match: {:.1%}".format(
            df.filter(df["nts_hh_id"].is_not_null()).unique("id").shape[0]
            / df.unique("id").shape[0]
        )
    )
    print(
        "% of people with any travel day: {:.1%}".format(
            df.filter(pl.col("TravDay").is_not_null()).group_by("id").all().shape[0]
            / df.unique("id").shape[0],
        )
    )
    print(
        "% of people with any weekday travel day: {:.1%}".format(
            df.group_by("id")
            .all()
            .filter(
                pl.lit(1).is_in(pl.col("TravDay"))
                | pl.lit(2).is_in(pl.col("TravDay"))
                | pl.lit(3).is_in(pl.col("TravDay"))
                | pl.lit(4).is_in(pl.col("TravDay"))
                | pl.lit(5).is_in(pl.col("TravDay"))
            )
            .shape[0]
            / df.unique("id").shape[0],
        )
    )
    print(
        "% of people with a travel day of {}: {:.1%}".format(
            trav_day,
            df.group_by("id").all().filter(pl.lit(3).is_in(pl.col("TravDay"))).shape[0]
            / df.unique("id").shape[0],
        )
    )

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
        # .filter(pl.col("Place of work indicator (4 categories) code").is_in([1, 3]))
        # Only travelling to work, no working from home
        .filter(pl.col("Place of work indicator (4 categories) code").is_in([3]))
        .rename({"Count": "census_count"})
        .group_by(["ozone", "dzone"])
        .sum()
        .select(
            [
                "ozone",
                "dzone",
                (pl.col("census_count") * scaling).cast(pl.Int64),
            ]
        )
        .collect()
    )

    # For no normalization
    both_counts = acbm_matrix.join(census_matrix, on=["ozone", "dzone"])

    print(both_counts)
    print("\nStats for counts...")
    print_stats(both_counts, "census_count", "acbm_count")

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

    print(both_norms)
    print("\nStats for proportions (by total origin)...")
    print_stats(both_norms, "census_norm", "acbm_norm")

    # print stats work
    print_stats_work(config)


if __name__ == "__main__":
    main()
