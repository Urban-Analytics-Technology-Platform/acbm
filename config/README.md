The config.toml file has an explanation for each parameter. You can copy the toml file, give it a name that is relevant to your project, and modify the parameters as needed. An example toml file, with an explanation of parameters is shown below. You can also find the latest version of the example config file in the [base.toml](https://github.com/Urban-Analytics-Technology-Platform/acbm/tree/main/config/base.toml) file in the config directory.

``` toml
[parameters]
seed = 0
region = "leeds"            # this is used to query poi data from osm and to load in SPC data
number_of_households = 5000 # how many people from the SPC do we want to run the model for? Comment out if you want to run the analysis on the entire SPC populaiton
zone_id = "OA21CD"          # "OA21CD": OA level, "MSOA11CD": MSOA level
travel_times = true         # Only set to true if you have travel time matrix at the level specified in boundary_geography
boundary_geography = "OA"   # boundary geography to use for the analysis
# NTS years to use
nts_years = [2019, 2021, 2022]
# NTS regions to use: the values here correspond categories for "PSUStatsReg_B01ID" in the NTS
nts_regions = [
    "Northern, Metropolitan",
    # "Northern, Non-metropolitan",
    "Yorkshire / Humberside, Metropolitan",
    # "Yorkshire / Humberside, Non-metropolitan",
    "East Midlands",
    "East Anglia",
    "South East (excluding London Boroughs)",
    # "London Boroughs",
    "South West",
    "West Midlands, Metropolitan",
    # "West Midlands, Non-metropolitan",
    "North West, Metropolitan",
    # "North West, Non-metropolitan",
]
# nts days of the week to use
# 1: Monday, 2: Tuesday, 3: Wednesday, 4: Thursday, 5: Friday, 6: Saturday, 7: Sunday
nts_days_of_week = [3]
# what crs do we want the output to be in? (just add the number, e.g. 3857)
output_crs = 3857
# tolerance_work: the proportion difference allowed for determining feasible zones for work
tolerance_work = 0.3
# tolerance_edu: the proportion difference allowed for determining feasible zones for education
tolerance_edu = 0.3
# common_household_day:
# - true: whether to only have households where all individuals have at least one day of the week in
#         common for travel.
# - false: households are kept only where all individuals have a travel day in NTS days of the week
common_household_day = true
# part_time_work_prob: this float scales the commuting flows since the census question is about
# main place of work and on a given day there is an average probability that a person is not at
# this main place of work. This assumes `use_percentages = false` in the [work_assignment] section
# and has no effect otherwise.
# See: https://github.com/Urban-Analytics-Technology-Platform/acbm/pull/90#issue-2823864300
# TODO: consider renaming this parameter and moving it to the [work_assignment] section
#       see: https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/99
part_time_work_prob = 0.7
n_processes = 6 # number of processes to use for parallel processing, excluded from serialization

[matching]
# for optional and required columns, see the [iterative_match_categorical](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/ca181c54d7484ebe44706ff4b43c26286b22aceb/src/acbm/matching.py#L110) function
# Do not add any column not listed below. You can only move a column from optional to require (or
# vise versa)
required_columns = ["number_adults", "number_children"]
optional_columns = [
    "number_cars",
    "num_pension_age",
    "rural_urban_2_categories",
    "employment_status",
    "tenure_status",
]
n_matches = 10 # What is the maximum number of NTS matches we want for each SPC household?

[feasible_assignment]
# `actual_distance = distance * (1 + ((detour_factor - 1) * np.exp(-decay_rate * distance)))`
#
# `detour factor` when converting Euclidean distance to actual travel distance
detour_factor = 1.56

# `decay rate` is the inverse of the distance (in units of the data, e.g. metres) at which the
# scaling from using the detour factor to Euclidean distance reduces by `exp(−1)`.
#
# 0.0001 is a good value for Leeds when units are metres, choice of decay_rate can be explored in an
# [interactive plot](https://www.wolframalpha.com/input?i=plot+exp%28-0.0001x%29+from+x%3D0+to+x%3D50000)
decay_rate = 0.0001

[work_assignment]
commute_level = "OA"
# if true, optimization problem will try to minimize percentage difference at OD level (not absolute
# numbers). Recommended to set it to true.
use_percentages = true
# weight for max deviation over all commuting flows for combined objective
weight_max_dev = 0.2
# weight for total deviation over all commuting flows for combined objective
weight_total_dev = 0.8
# maximum number of feasible zones to include in the optimization problem (less zones makes problem
# smaller - so faster, but at the cost of a better solution)
max_zones = 8

[secondary_assignment]
# Probablity of choosing a secondary zone. Same idea as a gravity model. We use floor_space / distance^n, where n is the power value used here
# See here to understand how this probability matrix is used https://github.com/arup-group/pam/blob/main/examples/17_advanced_discretionary_locations.ipynb
# Different values for the exponent can be tried to match the NTS data. Validation plots produced enable comparison of model distance distribution (by trip purpose) with the NTS
visit_probability_power = 2.0  # Default power value

[postprocessing]
pam_jitter = 30
pam_min_duration = 10
# for get_pt_subscription: everyone above this age has a subscription (pensioners get free travel)
# TODO: more sophisticated approach
pt_subscription_age = 66
# to define if a person is a student:
# eveyone below this age is a student
student_age_base = 16
# everyone below this age that has at least one "education" activity is a student
student_age_upper = 30
# eveyone who uses one of the modes below is classified as a passenger (isPassenger = True)
modes_passenger =  ['car_passenger', 'taxi']
# yearly state pension: for getting hhlIncome of pensioners
state_pension = 11502

```
