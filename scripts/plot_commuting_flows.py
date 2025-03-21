import os

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from screeninfo import get_monitors

import acbm
from acbm.config import load_config


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


def plot_all(study_area, df, column, ax, vmin=None, vmax=None, cmap="viridis"):
    legend_kwds = {"shrink": 0.3}
    print(df)
    df.plot(
        ax=ax,
        column=column,
        legend=True,
        cmap=cmap,
        legend_kwds=legend_kwds,
        vmin=vmin,
        vmax=vmax,
    )
    # centroid = study_area[study_area["MSOA21CD"].eq(origin)]["geometry"].centroid
    # centroid.plot(ax=ax, c="red", marker="x")
    study_area.plot(facecolor="none", edgecolor="black", lw=0.2, ax=ax)

    # 20km/h in car
    # (x, y) = (centroid.x.to_numpy()[0], centroid.y.to_numpy()[0])
    # ax.add_patch(plt.Circle((x, y), 20000, ec="grey", fill=False, ls=":"))
    ax.set_title(column)


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
@click.option(
    "--use-percentages",
    is_flag=True,
    default=False,
)
@click.option("--scaling", type=float, default=1.0, required=False)
def main(
    id: str | None, config_file: str | None, use_percentages: bool, scaling: float
):
    if id is not None and config_file is not None:
        print("Specify one of 'id' and 'config-file'.")
        exit(1)
    pd.options.mode.copy_on_write = True
    os.chdir(acbm.root_path)
    config = (
        load_config(f"data/outputs/{id}/config.toml")
        if config_file is None
        else load_config(config_file)
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
        # Only travelling to work, no working from home
        .filter(pl.col("Place of work indicator (4 categories) code").is_in([3]))
        .rename({"Count": "census_count"})
        .group_by(["ozone", "dzone"])
        .sum()
        .select(["ozone", "dzone", pl.col("census_count") * scaling])
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
    both_norms = both_norms.with_columns(
        (
            (both_norms["acbm_norm"] - both_norms["census_norm"])
            / both_norms["census_norm"]
        ).alias("error")
    )
    both_norms = both_norms.with_columns(
        (both_norms["acbm_norm"] - both_norms["census_norm"]).alias("acbm - census")
    )

    travel_time_estimates = pl.read_parquet(config.travel_times_estimates_filepath)

    # Use counts or percentages depending on CLI flag
    df = (both_norms if use_percentages else both_counts).join(
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

    # Get monitor information
    monitors = get_monitors()
    print("Available monitors:")
    print(monitors)
    monitor = monitors[0]

    # Plot difference in MSOA origins
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Difference in origin zones", size="medium")
    # Keep the grid for scale
    # [ax.axis("off") for ax in axs]

    manager = plt.get_current_fig_manager()
    # Resizing as wide fits better for this plot
    manager.resize(monitor.width * 5, monitor.height)

    sub = gpd.GeoDataFrame(
        df.to_pandas()
        .groupby("ozone")
        .sum()
        .drop(columns=["dzone"])
        .merge(study_area, left_on="ozone", right_on="MSOA21CD", how="inner")
        .rename(columns={"MSOA21CD": "ozone"})
        # .reset_index()
    )

    plot_all(study_area, sub, "census_count", ax=axs[0])
    plot_all(study_area, sub, "acbm_count", ax=axs[1])
    plot_all(
        study_area,
        sub,
        "acbm - census",
        ax=axs[2],
        vmin=-get_abs_max(sub, "acbm - census"),
        vmax=get_abs_max(sub, "acbm - census"),
        cmap="RdBu",
    )
    plt.show()

    gdf = gpd.GeoDataFrame(
        df.to_pandas()
        .merge(study_area, left_on="dzone", right_on="MSOA21CD", how="left")
        .drop(columns="MSOA21CD")
    )

    for _i, (origin, sub) in enumerate(gdf.groupby("ozone")):
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
