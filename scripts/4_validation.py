import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import acbm
from acbm.logger_config import validation_logger as logger
from acbm.validating.plots import plot_comparison, plot_intrazonal_trips
from acbm.validating.utils import calculate_od_distances, process_sequences

# ----- Folder for validation plots

logger.info("1. Creating folder for validation plots")

validation_plots_path = acbm.root_path / "data/processed/plots/validation"
os.makedirs(validation_plots_path, exist_ok=True)


# ----- Reading in the data

logger.info("2. Reading in the data")

# NTS data
legs_nts = pd.read_parquet(
    acbm.root_path / "data/external/nts/filtered/nts_trips.parquet"
)

legs_nts = legs_nts[legs_nts["TravDay"] == 3]

# Model outputs
legs_acbm = pd.read_csv(acbm.root_path / "data/processed/activities_pam/legs.csv")
legs_acbm_geo = pd.read_parquet(
    acbm.root_path / "data/processed/activities_pam/legs_with_locations.parquet"
)

# ----- Preproccessing the data

logger.info("3a. Preprocessing: renaming columns")

# rename origin activity and destination activity columns
legs_acbm = legs_acbm.rename(
    columns={"origin activity": "oact", "destination activity": "dact"}
)
legs_acbm_geo = legs_acbm_geo.rename(
    columns={"origin activity": "oact", "destination activity": "dact"}
)

# rename distance column in NTS
legs_nts = legs_nts.rename(columns={"TripDisIncSW": "distance"})

logger.info("3b. Preprocessing: Edit distance column in NTS")

# convert legs_nts["distance"] from miles to km
legs_nts["distance"] = legs_nts["distance"] * 1.60934

logger.info("3c. Preprocessing: Adding hour of day column")

# acbm - tst is in datetime format
# Convert tst to datetime format and extract the hour component in one step
legs_acbm["tst_hour"] = legs_acbm["tst"].apply(lambda x: pd.to_datetime(x).hour)
legs_acbm["tet_hour"] = legs_acbm["tet"].apply(lambda x: pd.to_datetime(x).hour)

# nts - tst is in minutes
# Convert legs_nts["tst"] from minutes to hours
legs_nts["tst_hour"] = legs_nts["tst"] // 60
legs_nts["tet_hour"] = legs_nts["tet"] // 60

logger.info("3d. Preprocessing: Adding primary / secondary trip column")

# Define the conditions for primary trips
conditions_primary = (
    (legs_acbm["oact"] == "home") & (legs_acbm["dact"].isin(["work", "education"]))
) | ((legs_acbm["oact"].isin(["work", "education"])) & (legs_acbm["dact"] == "home"))

# Add the trip_type column
legs_acbm["trip_type"] = np.where(conditions_primary, "primary", "secondary")

logger.info("3e. Preprocessing: Abbreviating column values for trip purpose")

# Mapping dictionary
activity_mapping = {
    "home": "h",
    "other": "o",
    "escort": "e",
    "work": "w",
    "shop": "sh",
    "visit": "v",
    "education": "edu",
    "medical": "m",
}

legs_acbm["oact_abr"] = legs_acbm["oact"].replace(activity_mapping)
legs_acbm["dact_abr"] = legs_acbm["dact"].replace(activity_mapping)

legs_nts["oact_abr"] = legs_nts["oact"].replace(activity_mapping)
legs_nts["dact_abr"] = legs_nts["dact"].replace(activity_mapping)


# ----- Validation Plots

logger.info("4. Validation plots")

logger.info("4a.1 Validation (Matching) - Trip Purpose")

# Get number of trips by mode for legs_nts, and legs_acbm, and plot a comparative bar plot
# NTS
purpose_nts = legs_nts.groupby("dact").size().reset_index(name="count")
purpose_nts["source"] = "nts"

# ACBM
purpose_acbm = legs_acbm.groupby("dact").size().reset_index(name="count")
purpose_acbm["source"] = "acbm"

# Combine the data
purpose_compare = pd.concat([purpose_nts, purpose_acbm])

# Calculate the percentage of trips for each mode within each source
purpose_compare["percentage"] = purpose_compare.groupby("source")["count"].transform(
    lambda x: (x / x.sum()) * 100
)


sns.barplot(data=purpose_compare, x="dact", y="percentage", hue="source")
plt.xlabel("Trip purpose")
plt.ylabel("Percentage of total trips")
plt.title("Percentage of Trips by Purpose for NTS and ACBM")
# plt.show()

# Save the plot
plt.tight_layout()
plt.savefig(validation_plots_path / "1_matching_trip_purpose.png")

logger.info("4a.2 Validation (Matching) - Trip Mode")

# Get number of trips by mode for legs_nts, and legs_acbm, and plot a comparative bar plot
# NTS
modeshare_nts = legs_nts.groupby("mode").size().reset_index(name="count")
modeshare_nts["source"] = "nts"

# ACBM
modeshare_acbm = legs_acbm.groupby("mode").size().reset_index(name="count")
modeshare_acbm["source"] = "acbm"

# Combine the data
modeshare_compare = pd.concat([modeshare_nts, modeshare_acbm])
# Calculate the percentage of trips for each mode within each source
modeshare_compare["percentage"] = modeshare_compare.groupby("source")[
    "count"
].transform(lambda x: (x / x.sum()) * 100)

# Plot
sns.barplot(data=modeshare_compare, x="mode", y="percentage", hue="source")
plt.ylabel("Percentage of total trips")
plt.title("Percentage of Trips by Mode for NTS and ACBM")
# plt.show()

# Save the plot
plt.tight_layout()
plt.savefig(validation_plots_path / "2_matching_trip_mode.png")

logger.info("4a.3 Validation (Matching) - time of day")

# Plot aggregate
plot_comparison(
    legs_acbm,
    legs_nts,
    value_column="tst_hour",
    max_y_value=20,
    plot_type="time",
    figsize=(10, 5),
    plot_mode="aggregate",
    save_path=validation_plots_path / "3a_matching_time_of_day_aggregate.png",
)

# Plot facet
plot_comparison(
    legs_acbm,
    legs_nts,
    value_column="tst_hour",
    max_y_value=70,
    plot_type="time",
    plot_mode="facet",
    save_path=validation_plots_path / "3b_matching_time_of_day_facet.png",
)


logger.info("4a.4 Validation (Matching) - Activity Sequences")

# Process the sequences for ACBM and NTS data

sequence_nts = process_sequences(
    df=legs_nts,
    pid_col="IndividualID",
    seq_col="seq",
    origin_activity_col="oact_abr",
    destination_activity_col="dact_abr",
    suffix="nts",
)

sequence_acbm = process_sequences(
    df=legs_acbm,
    pid_col="pid",
    seq_col="seq",
    origin_activity_col="oact_abr",
    destination_activity_col="dact_abr",
    suffix="acbm",
)


# Plot the comparison

# join the two dataframes by 'activity_sequence'
sequence_nts_acbm = sequence_nts.merge(
    sequence_acbm, on="activity_sequence", how="inner"
).sort_values(by="count_nts", ascending=False)

# Get % contribution of each unique activity sequence
sequence_nts_acbm["count_nts"] = (
    sequence_nts_acbm["count_nts"] / sequence_nts_acbm["count_nts"].sum() * 100
)
sequence_nts_acbm["count_acbm"] = (
    sequence_nts_acbm["count_acbm"] / sequence_nts_acbm["count_acbm"].sum() * 100
)

# Filter rows where both count columns are bigger than x %
x = 0.35

sequence_nts_acbm_filtered = sequence_nts_acbm[
    (sequence_nts_acbm["count_nts"] > x) & (sequence_nts_acbm["count_acbm"] > x)
]

fig, ax = plt.subplots(figsize=(10, 6))

sequence_nts_acbm_filtered.plot(
    x="activity_sequence", y=["count_nts", "count_acbm"], kind="bar", ax=ax
)

plt.ylabel("Percentage of total trips")
plt.title("Comparison of Activity Sequences between NTS and ACBM")

# Add the color legend to the plot
plt.legend(["NTS", "ACBM"], loc="upper right")
# Generate custom legend
legend_labels = [f"{abbr} = {full}" for abbr, full in activity_mapping.items()]
custom_legend = " | ".join(legend_labels)
# Add the custom legend below the chart
plt.figtext(
    0.5, -0.2, custom_legend, wrap=True, horizontalalignment="center", fontsize=12
)

# plt.show()

# Save the plot
plt.tight_layout()
plt.savefig(validation_plots_path / "4_matching_activity_sequences.png")


logger.info("4b. Validation (Assigning) - Trip Distance")

# Apply the function to legs_acbm_geo
legs_acbm_geo = calculate_od_distances(
    df=legs_acbm_geo,
    start_wkt_col="start_location_geometry_wkt",
    end_wkt_col="end_location_geometry_wkt",
)


# Plot: Aggregate
plot_comparison(
    legs_acbm_geo,
    legs_nts,
    value_column="distance",
    bin_size=2,
    value_threshold=50,
    max_y_value=30,
    figsize=(10, 5),
    plot_type="distance",
    plot_mode="aggregate",
    save_path=validation_plots_path / "5a_assigning_distance_aggregate.png",
)

# Plot: Facet by activity_type
plot_comparison(
    legs_acbm_geo,
    legs_nts,
    value_column="distance",
    bin_size=2,
    value_threshold=50,
    max_y_value=30,
    plot_type="distance",
    plot_mode="facet",
    save_path=validation_plots_path / "5b_assigning_distance_facet.png",
)


logger.info("4c.1 Validation (Assigning) - Intrazonal Activities")

# -- Plot: by trip purpose
# Calculate the percentage of intrazonal trips for each unique purpose

plot_intrazonal_trips(
    legs_acbm,
    validation_plots_path=validation_plots_path,
    plot_type="purp",
    plot_name="6a_assigning_intrazonal_purp.png",
)

# -- Plot: By OD pair
# Calculate the percentage of intrazonal trips for each unique OD combination
# (e.g. home - work)

plot_intrazonal_trips(
    legs_acbm,
    validation_plots_path=validation_plots_path,
    plot_type="od",
    plot_name="6b_assigning_intrazonal_od.png",
)
