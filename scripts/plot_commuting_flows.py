import os

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from screeninfo import get_monitors

import acbm
from acbm.config import load_config


@click.command()
@click.option(
    "--id",
    type=str,
    required=False,
    default=None,
)
@click.option(
    "--config-file",
    type=str,
    default=None,
    required=False,
)
def main(id: str | None, config_file: str | None):
    if id is not None and config_file is not None:
        print("Specify one of 'id' and 'config-file'.")
        exit(1)
    pd.options.mode.copy_on_write = True
    os.chdir(acbm.root_path)
    if config_file is None:
        config = load_config(f"data/outputs/{id}/config.toml")
    else:
        config = load_config(config_file)

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
        # Work From Home (1) and Travel to work (3)
        # .filter(pl.col("Place of work indicator (4 categories) code").is_in([1, 3]))
        # Only travelling to work, no working from home
        .filter(pl.col("Place of work indicator (4 categories) code").is_in([3]))
        .rename({"Count": "census_count"})
        .group_by(["ozone", "dzone"])
        .sum()
        .select(["ozone", "dzone", "census_count"])
        .collect()
    )

    # For no normalization
    both_counts = acbm_matrix.join(census_matrix, on=["ozone", "dzone"])
    both_counts = both_counts.with_columns(
        (
            (both_counts["acbm_count"] - both_counts["census_count"])
            / both_counts["census_count"]
        ).alias("error")
    )
    both_counts = both_counts.with_columns(
        (both_counts["acbm_count"] - both_counts["census_count"]).alias("acbm - census")
    )

    travel_time_estimates = pl.read_parquet(config.travel_times_estimates_filepath)
    df = both_counts.join(
        travel_time_estimates.select(
            ["MSOA21CD_from", "MSOA21CD_to", "distance"]
        ).unique(),
        left_on=["ozone", "dzone"],
        right_on=["MSOA21CD_from", "MSOA21CD_to"],
        how="left",
        coalesce=True,
    )

    df.group_by("ozone").agg(
        [pl.col("error"), pl.col("acbm - census"), pl.col("dzone"), pl.col("distance")]
    )

    study_area = gpd.read_file(config.study_area_filepath)
    gdf = gpd.GeoDataFrame(
        df.to_pandas()
        .merge(study_area, left_on="dzone", right_on="MSOA21CD", how="left")
        .drop(columns="MSOA21CD")
    )

    def get_abs_max(df_, column) -> int | float:
        return abs(df_[column].to_numpy()).max()

    def plot_column(
        study_area, df, column, ax, origin, vmin=None, vmax=None, cmap="viridis"
    ):
        legend_kwds = {"shrink": 0.3}
        df.plot(
            ax=ax,
            column=column,
            legend=True,
            cmap=cmap,
            legend_kwds=legend_kwds,
            vmin=vmin,
            vmax=vmax,
        )
        centroid = study_area[study_area["MSOA21CD"].eq(origin)]["geometry"].centroid
        centroid.plot(ax=ax, c="red", marker="x")
        study_area.plot(facecolor="none", edgecolor="black", lw=0.2, ax=ax)

        # 20km/h in car
        (x, y) = (centroid.x.to_numpy()[0], centroid.y.to_numpy()[0])
        ax.add_patch(plt.Circle((x, y), 20000, ec="grey", fill=False, ls=":"))
        ax.set_title(column)

    # Get monitor information
    monitors = get_monitors()
    print("Available monitors:")
    print(monitors)
    monitor = monitors[0]
    for _i, (origin, sub) in enumerate(gdf.groupby("ozone")):
        # fig, axs = plt.subplots(1, 3, figsize=(20, 16))
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(
            f"Counts from origin zone '{origin}' to destination zones", size="medium"
        )
        # Keep the grid for scale
        # [ax.axis("off") for ax in axs]

        manager = plt.get_current_fig_manager()
        # Resizing as wide fits better for this plot
        manager.resize(monitor.width * 5, monitor.height)

        plot_column(study_area, sub, "census_count", origin=origin, ax=axs[0])
        plot_column(study_area, sub, "acbm_count", origin=origin, ax=axs[1])
        plot_column(
            study_area,
            sub,
            "acbm - census",
            origin=origin,
            ax=axs[2],
            vmin=-get_abs_max(sub, "acbm - census"),
            vmax=get_abs_max(sub, "acbm - census"),
            cmap="RdBu",
        )
        plt.show()


if __name__ == "__main__":
    main()
