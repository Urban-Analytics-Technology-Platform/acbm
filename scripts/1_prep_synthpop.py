#!/usr/bin/env python
# coding: utf-8

# ## Preparing the Synthetic Population

# We will use the spc package for our synthetic population. To add it as a dependancy in this virtual environment, I ran `poetry add git+https://github.com/alan-turing-institute/uatk-spc.git@55-output-formats-python#subdirectory=python`. The branch may change if the python package is merged into the main spc branch.

# import json
import pandas as pd

# https://github.com/alan-turing-institute/uatk-spc/blob/55-output-formats-python/python/examples/spc_builder_example.ipynb
from uatk_spc.builder import Builder


# ### Loading in the SPC synthetic population
#
# I use the code in the `Quickstart` [here](https://github.com/alan-turing-institute/uatk-spc/blob/55-output-formats-python/python/README.md) to get a parquet file and convert it to JSON.
#
# You have two options:
#
#
# 1- Slow and memory-hungry: Download the pbf file directly from [here](https://alan-turing-institute.github.io/uatk-spc/using_england_outputs.html) and load in the pbf file with the python package
#
# 2- Faster: Covert the pbf file to parquet, and then load it using the python package. To convert to parquet, you need to:
#
# a. clone the [uatk-spc](https://github.com/alan-turing-institute/uatk-spc/tree/main/docs)
#
# b. Run `cargo run --release -- --rng-seed 0 --flat-output config/England/west-yorkshire.txt --year 2020`  and replace `west-yorkshire` and `2020` with your preferred option
#

# Pick a region with SPC output saved
path = "../data/external/spc_output/raw/"
region = "west-yorkshire"


# #### People and household data

# add people and households
spc_people_hh = (
    Builder(path, region, backend="pandas", input_type="parquet")
    .add_households()
    .unnest(["health", "employment", "details", "demographics"], rsuffix="_household")
    .build()
)

spc_people_hh.head(5)


# save the output
spc_people_hh.to_parquet("../data/external/spc_output/" + region + "_people_hh.parquet")


spc_people_hh["salary_yearly"].hist(bins=100)


# plt.show()


spc_people_hh["salary_yearly"].unique()


# #### People and time-use data

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
spc_people_tu.head()


# save the output
spc_people_tu.write_parquet(
    "../data/external/spc_output/" + region + "_people_tu.parquet"
)
