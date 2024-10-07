from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_comparison(
    legs_acbm: pd.DataFrame,
    legs_nts: pd.DataFrame,
    activity_column: str = "dact",
    value_column: str = "distance",
    bin_size: Optional[int] = None,
    num_cols: int = 4,
    max_y_value: Optional[int] = None,
    figsize: Tuple[int, int] = (20, 5),
    value_threshold: Optional[int] = None,
    plot_type: str = "distance",
    plot_mode: str = "facet",
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a comparison of data for different activity types. It can plot either distance
    or time data, and can plot the data in either facet or aggregate mode (facet mode
    creates a separate plot for each activity type)

    Parameters
    ----------
    legs_acbm : pd.DataFrame
        DataFrame containing the ACBM data.
    legs_nts : pd.DataFrame
        DataFrame containing the NTS data.
    activity_column : str, optional
        The column name for the activity types. Default is 'dact'.
    value_column : str, optional
        The column name for the values to be plotted. Default is 'distance'.
    bin_size : int, optional
        The size of the bins for rounding values. Default is None.
    num_cols : int, optional
        The number of columns for the subplots. Default is 4.
    max_y_value : int, optional
        The maximum value for the y-axis. Default is None.
    figsize : tuple of int, optional
        The size of the figure. Default is (20, 5).
    value_threshold : int, optional
        The maximum value for the x-axis. Default is None.
    plot_type : str, optional
        The type of plot ('distance' or 'time'). Default is 'distance'.
    plot_mode : str, optional
        The mode of plot ('facet' or 'aggregate'). Default is 'facet'.
    save_path : str, optional
        The file path to save the plot. Default is None.

    Returns
    -------
    None
        This function generates and displays a plot but does not return any value.
    """
    if plot_type not in ["distance", "time"]:
        msg = "plot_type must be either 'distance' or 'time'"
        raise ValueError(msg)

    if plot_mode not in ["facet", "aggregate"]:
        msg = "plot_mode must be either 'facet' or 'aggregate'"
        raise ValueError(msg)

    if plot_type == "distance" and bin_size is None:
        msg = "bin_size must be provided when plot_type is 'distance'"
        raise ValueError(msg)

    if plot_type == "distance":
        # Create binned column for distance
        legs_acbm["value_binned"] = (
            legs_acbm[value_column] / bin_size
        ).round() * bin_size
        legs_nts["value_binned"] = (
            legs_nts[value_column] / bin_size
        ).round() * bin_size

        # Define the bins
        max_value_data = max(
            legs_acbm["value_binned"].max(), legs_nts["value_binned"].max()
        )
        if value_threshold is not None:
            max_value = min(value_threshold, max_value_data)
        else:
            max_value = max_value_data
        bins = range(0, int(max_value) + bin_size, bin_size)
    else:
        legs_acbm["value_binned"] = legs_acbm[value_column]
        legs_nts["value_binned"] = legs_nts[value_column]
        bins = range(25)  # For time of day, we use 24 bins (0-24 hours)

    if plot_mode == "facet":
        # Get unique activity values
        unique_activity_values = legs_acbm[activity_column].unique()

        # Create a figure with subplots
        num_plots = len(unique_activity_values)
        num_rows = (num_plots + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(figsize[0], num_rows * figsize[1]), sharey=True
        )
        axes = axes.flatten()

        # Iterate over unique activity values and create plots
        for i, activity_value in enumerate(unique_activity_values):
            ax = axes[i]
            acbm_data = legs_acbm[legs_acbm[activity_column] == activity_value]
            nts_data = legs_nts[legs_nts[activity_column] == activity_value]

            # Plot histogram for acbm_data
            sns.histplot(
                acbm_data["value_binned"],
                bins=bins,
                kde=False,
                discrete=True,
                stat="percent",
                ax=ax,
                label="ACBM",
            )
            ax.set_title(f"{activity_column}: {activity_value}", fontsize=16)
            ax.tick_params(
                axis="x", rotation=45, labelsize=12
            )  # Rotate x-axis labels by 45 degrees

            # Set x-axis limits based on max_value
            if plot_type == "distance":
                ax.set_xlim(0, max_value)
            else:
                ax.set_xlim(0, 24)  # For time of day, x-axis is 0-24

            # Set y-axis limits based on max_y_value
            if max_y_value:
                ax.set_ylim(0, max_y_value)

            # Add x-axis ticks and labels for each bar
            if plot_type == "distance":
                ax.set_xticks(bins)
                ax.set_xticklabels([str(bin) for bin in bins], rotation=45, ha="right")
            else:
                ax.set_xticks(
                    range(25)
                )  # For time of day, set x-ticks to represent each hour of the day

            # Remove individual subplot labels
            ax.set_xlabel("")
            ax.set_ylabel("")  # Remove y-axis label for individual plots

            # Add dots to represent the percentage values of legs_nts["value_binned"]
            nts_value_counts = (
                nts_data["value_binned"].value_counts(normalize=True).sort_index() * 100
            )
            if not nts_value_counts.empty:
                bin_centers = nts_value_counts.index
                ax.plot(
                    bin_centers, nts_value_counts.values, "ro", label="NTS"
                )  # 'ro' means red dot

        # Remove any empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Add a main legend at the bottom
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05)
        )

        # Add a main title to the figure
        if plot_type == "distance":
            fig.suptitle(
                "Comparison of Trip Distance for Different Activity Types", fontsize=20
            )
            fig.text(0.5, 0.02, "Distance (km)", ha="center", fontsize=14)
        else:
            fig.suptitle(
                "Comparison of Trip Start Time for Different activity Types",
                fontsize=20,
            )
            fig.text(0.5, 0.02, "Hour of Day", ha="center", fontsize=14)

        # Add a single centered y-label
        fig.text(
            0.02,
            0.5,
            "Percentage of Trips",
            va="center",
            rotation="vertical",
            fontsize=14,
        )

        # Adjust layout to make room for the main title and labels
        plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])

    elif plot_mode == "aggregate":
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram for acbm_data
        sns.histplot(
            legs_acbm["value_binned"],
            bins=bins,
            kde=False,
            discrete=True,
            stat="percent",
            ax=ax,
            label="ACBM",
        )
        # ax.set_title('Aggregate Comparison', fontsize=16)
        ax.tick_params(
            axis="x", rotation=45, labelsize=12
        )  # Rotate x-axis labels by 45 degrees

        # Set x-axis limits based on max_value
        if plot_type == "distance":
            ax.set_xlim(0, max_value)
            ax.set_xlabel("Distance (km)")
        else:
            ax.set_xlim(0, 24)  # For time of day, x-axis is 0-24
            ax.set_xlabel("Hour of Day")

        # Set y-axis limits based on max_y_value
        if max_y_value:
            ax.set_ylim(0, max_y_value)

        # Add x-axis ticks and labels for each bar
        if plot_type == "distance":
            ax.set_xticks(bins)
            ax.set_xticklabels([str(bin) for bin in bins], rotation=45, ha="right")
        else:
            ax.set_xticks(
                range(25)
            )  # For time of day, set x-ticks to represent each hour of the day

        # Add dots to represent the percentage values of legs_nts["value_binned"]
        nts_value_counts = (
            legs_nts["value_binned"].value_counts(normalize=True).sort_index() * 100
        )
        if not nts_value_counts.empty:
            bin_centers = nts_value_counts.index
            ax.plot(
                bin_centers, nts_value_counts.values, "ro", label="NTS"
            )  # 'ro' means red dot

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right")

        # Add a main title to the figure
        if plot_type == "distance":
            fig.suptitle("Comparison of Trip Distance", fontsize=20)
        else:
            fig.suptitle("Comparison of Trip Start Time", fontsize=20)

        # Add a single centered y-label
        ax.set_ylabel("Percentage of Trips")  # Set y-axis label for aggregate plot

        # Adjust layout to make room for the main title and labels
        plt.tight_layout(rect=[0.03, 0.05, 1, 0.95])

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    # Show the plot
    # plt.show()


# Intrazonal trips


def _calculate_intrazonal_counts(
    legs_acbm: pd.DataFrame, group_by_columns: list
) -> pd.DataFrame:
    """
    Calculate total and intrazonal counts and merge them.

    Parameters
    ----------
    legs_acbm : pd.DataFrame
        DataFrame containing the ACBM data.
    group_by_columns : list
        List of columns to group by.

    Returns
    -------
    pd.DataFrame
        DataFrame with total and intrazonal counts and percentages.
    """
    legs_acbm["intrazonal"] = legs_acbm["ozone"] == legs_acbm["dzone"]

    # Total number of trips per group
    total_counts = (
        legs_acbm.groupby(group_by_columns).size().reset_index(name="total_count")
    )

    # Filter the DataFrame to include only rows where intrazonal_trips is TRUE
    intrazonal_trips_true = legs_acbm[legs_acbm["intrazonal"]]

    # Total number of intrazonal trips per group
    intrazonal_counts = (
        intrazonal_trips_true.groupby(group_by_columns)
        .size()
        .reset_index(name="intrazonal_count")
    )

    # Merge the two DataFrames and calculate intrazonal %
    merged_counts = pd.merge(
        total_counts, intrazonal_counts, on=group_by_columns, how="left"
    )
    # Fill NaN values with 0 (in case there are groups with no intrazonal trips)
    merged_counts["intrazonal_count"] = merged_counts["intrazonal_count"].fillna(0)
    # Calculate the percentage of intrazonal trips
    merged_counts["percentage"] = (
        merged_counts["intrazonal_count"] / merged_counts["total_count"]
    ) * 100

    return merged_counts


def _plot_intrazonal_counts(
    merged_counts: pd.DataFrame,
    x_column: str,
    hue_column: Optional[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[Path] = None,
    plot_name: Optional[str] = None,
) -> None:
    """
    Plot the counts and optionally save the plot.

    Parameters
    ----------
    merged_counts : pd.DataFrame
        DataFrame with merged counts and percentages.
    x_column : str
        Column name for the x-axis.
    hue_column : Optional[str]
        Column name for the hue (optional).
    title : str
        Title of the plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    save_path : Optional[Path]
        Path to save the validation plot (optional).
    plot_name : str
        Name of the plot file to save.

    Returns
    -------
    None
    """
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    if hue_column:
        barplot = sns.barplot(
            data=merged_counts,
            x=x_column,
            y="percentage",
            hue=hue_column,
            palette="viridis",
        )
    else:
        barplot = sns.barplot(
            data=merged_counts,
            x=x_column,
            y="percentage",
            hue=x_column,
            palette="viridis",
            legend=False,
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45 if x_column == "purp" else 60)

    # Add text annotations above each bar
    for bar, row in zip(barplot.patches, merged_counts.iterrows()):
        barplot.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"({int(row[1]['total_count'])})",
            color="black",
            ha="center",
        )

    # Remove the top and right spines (box frame) from the plot
    barplot.spines["top"].set_visible(False)
    barplot.spines["right"].set_visible(False)

    # Add footnote on the right below the plot
    plt.figtext(0.95, -0.02, "(Total number of trips)", ha="right", fontsize=10)

    plt.tight_layout()
    # plt.show()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path / plot_name)


def plot_intrazonal_trips(
    legs_acbm: pd.DataFrame,
    validation_plots_path: Optional[Path] = None,
    plot_type: str = "od",
    plot_name: Optional[str] = None,
) -> None:
    """
    Plots the percentage of intrazonal trips per purpose or origin-destination pair.

    Parameters
    ----------
    legs_acbm : pd.DataFrame
        DataFrame containing the ACBM data.
    validation_plots_path : Optional[Path]
        Path to save the validation plot (optional).
    plot_type : str, optional
        The type of plot ('od' or 'purp'). Default is 'od'.

    Returns
    -------
    None
        This function generates and displays a plot but does not return any value.
    """
    if plot_type not in ["od", "purp"]:
        msg = "plot_type must be either 'od' or 'purp'"
        raise ValueError(msg)

    # Set default plot name based on plot_type if not provided
    if plot_name is None:
        plot_name = f"assigning_intrazonal_activities_{plot_type}.png"

    if plot_type == "od":
        # Add the trip_type column
        conditions_primary = (
            (legs_acbm["oact"] == "home")
            & (legs_acbm["dact"].isin(["work", "education"]))
        ) | (
            (legs_acbm["oact"].isin(["work", "education"]))
            & (legs_acbm["dact"] == "home")
        )
        legs_acbm["trip_type"] = np.where(conditions_primary, "primary", "secondary")

        # Create an od column to identify the origin-destination pairs
        legs_acbm["od"] = legs_acbm["oact"] + " - " + legs_acbm["dact"]

        # Calculate counts
        merged_counts = _calculate_intrazonal_counts(legs_acbm, ["od", "trip_type"])
        # Keep top 15 od pairs
        merged_counts = merged_counts.sort_values(
            by="total_count", ascending=False
        ).head(15)
        # Sort by percentage before plotting
        merged_counts = merged_counts.sort_values(by="percentage", ascending=False)

        # Plot counts
        _plot_intrazonal_counts(
            merged_counts,
            "od",
            "trip_type",
            "Percentage of Intrazonal Trips per Purpose",
            "Purpose",
            "Percentage of Trips that are Intrazonal",
            validation_plots_path,
            plot_name,
        )

    elif plot_type == "purp":
        # Calculate counts
        merged_counts = _calculate_intrazonal_counts(legs_acbm, ["purp"])
        # Sort by percentage before plotting
        merged_counts = merged_counts.sort_values(by="percentage", ascending=False)

        # Plot counts
        _plot_intrazonal_counts(
            merged_counts,
            "purp",
            None,
            "Percentage of Intrazonal Trips per Purpose",
            "Purpose",
            "Percentage of Trips that are Intrazonal",
            validation_plots_path,
            plot_name,
        )
