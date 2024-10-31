# acbm

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

A package to create activity-based models (for transport demand modelling)

- [acbm](#acbm)
  - [Motivation](#motivation)
  - [Contribution](#contribution)
  - [Installation](#installation)
  - [How to Run the Pipeline](#how-to-run-the-pipeline)
    - [Step 1: Prepare Data Inputs](#step-1-prepare-data-inputs)
    - [Step 2: Setup your config.toml file](#step-2-setup-your-configtoml-file)
    - [Step 3: Run the pipeline](#step-3-run-the-pipeline)
    - [Future Work](#future-work)
    - [Related Work](#related-work)
  - [Contributing](#contributing)
  - [License](#license)


## Motivation

## Contribution

## Installation

```bash
python -m pip install acbm
```

From source:
```bash
git clone https://github.com/alan-turing-institute/acbm
cd acbm
poetry install
```

## How to Run the Pipeline

The pipeline is a series of scripts that are run in sequence to generate the activity-based model. There are a few external datasets that are required. The data and config directories are structured as follows:

```md
├── config
│   ├── <your_config_1>.toml
│   ├── <your_config_2>.toml
├── data
│   ├── external
│   │   ├── boundaries
│   │   │   ├── MSOA_DEC_2021_EW_NC_v3.geojson
│   │   │   ├── oa_england.geojson
│   │   │   ├── study_area_zones.geojson
│   │   ├── census_2011_rural_urban.csv
│   │   ├── centroids
│   │   │   ├── LSOA_Dec_2011_PWC_in_England_and_Wales_2022.csv
│   │   │   └── Output_Areas_Dec_2011_PWC_2022.csv
│   │   ├── MSOA_2011_MSOA_2021_Lookup_for_England_and_Wales.csv
│   │   ├── nts
│   │   │   ├── filtered
│   │   │   │   ├── nts_households.parquet
│   │   │   │   ├── nts_individuals.parquet
│   │   │   │   └── nts_trips.parquet
│   │   │   └── UKDA-5340-tab
│   │   │       ├── 5340_file_information.rtf
│   │   │       ├── mrdoc
│   │   │       │   ├── excel
│   │   │       │   ├── pdf
│   │   │       │   ├── UKDA
│   │   │       │   └── ukda_data_dictionaries.zip
│   │   │       └── tab
│   │   │           ├── household_eul_2002-2022.tab
│   │   │           ├── individual_eul_2002-2022.tab
│   │   │           ├── psu_eul_2002-2022.tab
│   │   │           ├── trip_eul_2002-2022.tab
│   │   │           └── <other_nts_tables>.tab
│   │   ├── ODWP01EW_OA.zip
│   │   ├── ODWP15EW_MSOA_v1.zip
│   │   ├── spc_output
│   │   │   ├── <region>>_people_hh.parquet (Generated in Script 1)
│   │   │   ├── <region>>_people_tu.parquet (Generated in Script 1)
│   │   │   ├── raw
│   │   │   │   ├── <region>_households.parquet
│   │   │   │   ├── <region>_info_per_msoa.json
│   │   │   │   ├── <region>.pb
│   │   │   │   ├── <region>_people.parquet
│   │   │   │   ├── <region>_time_use_diaries.parquet
│   │   │   │   ├── <region>_venues.parquet
│   │   │   │   ├── README.md
│   │   └── travel_times
│   │       ├── oa
│   │       |   ├── travel_time_matrix.parquet
|   |       └── msoa
│   │           └── travel_time_matrix.parquet
│   ├── interim
│   │   ├── assigning (Generated in Script 3)
│   │   └── matching (Generated in Script 2)
│   └── processed
│       ├── acbm_<config_name>_<date>
│       │   ├── activities.csv
│       │   ├── households.csv
│       │   ├── legs.csv
│       │   ├── legs_with_locations.parquet
│       │   ├── people.csv
│       │   └── plans.xml
│       ├── plots
│       │   ├── assigning
│       │   └── validation
```

### Step 1: Prepare Data Inputs

You need to populate the data/external diectory with the required datasets. A guide on where to find / generate each dataset can be found in the [data/external/README.md]

### Step 2: Setup your config.toml file

You need to create a config file in the config directory. The config file is a toml file that contains the parameters for the pipeline. A guide on how to set up the config file can be found in the [config/README.md]

### Step 3: Run the pipeline

The scripts are listed in order of execution in the [scripts/run_pipeline.sh](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/main/scripts/run_pipeline.sh) bash file

You can run the pipeline by executing the following command in the terminal from the base directory:

```bash
bash ./scripts/run_pipeline.sh config/<your_config_file>.toml
```

where your config file is the file you created in Step 2.

### Future Work

### Related Work



## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [Apache license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/acbm/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/acbm/actions
[pypi-link]:                https://pypi.org/project/acbm/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/acbm
[pypi-version]:             https://img.shields.io/pypi/v/acbm
<!-- prettier-ignore-end -->
