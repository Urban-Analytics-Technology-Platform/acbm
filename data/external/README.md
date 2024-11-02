The folder contains all external datasets necessary to run the pipeline. Some can be downloaded, while others need to be generated. The README.md file in this folder provides a guide on where to find / generate each dataset.


## Folder Structure

The structure of the folder is as follows:

```md
.
├── data
│   ├── external
│   │   ├── boundaries
│   │   │   ├── MSOA_DEC_2021_EW_NC_v3.geojson
│   │   │   ├── oa_england.geojson
│   │   │   ├── study_area_zones.geojson
│   │   ├── census_2011_rural_urban.csv
│   │   ├── centroids
│   │   │   ├── LSOA_Dec_2011_PWC_in_England_and_Wales_2022.csv
│   │   │   ├── Output_Areas_Dec_2011_PWC_2022.csv
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
|   |   └── travel_times
|   |   │   │       ├── oa
|   |   │   │       |   ├── travel_time_matrix.parquet
|   |   |   |       └── msoa
|   |   │   │           └── travel_time_matrix.parquet
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

```

## Data Sources


`spc_output/`

Use the code in the `Quickstart` [here](https://github.com/alan-turing-institute/uatk-spc/blob/55-output-formats-python/python/README.md)
to get a parquet file and convert it to JSON.

You have two options:
1. Slow and memory-hungry: download the `.pb` file directly from [here](https://alan-turing-institute.github.io/uatk-spc/using_england_outputs.html)
    and load in the pbf file with the python package
2. Faster: Run SPC to generate parquet outputs, and then load using the SPC toolkit python package. To generate parquet, you need to:
    1. Clone [uatk-spc](https://github.com/alan-turing-institute/uatk-spc/tree/main/docs)
    2. Run:
        ```shell
        cargo run --release -- \
            --rng-seed 0 \
            --flat-output \
            --year 2020 \
            config/England/west-yorkshire.txt
        ```
        and replace `west-yorkshire` and `2020` with your preferred option.

`boundaries/`

- MSOA_DEC_2021_EW_NC_v3.geojson
- oa_england.geojson
- study_area_zones.geojson

`centroids/`

- LSOA_Dec_2011_PWC_in_England_and_Wales_2022.csv
- Output_Areas_Dec_2011_PWC_2022.csv

`nts/`

UKDA-5340-tab:
- Download the UKDA-5340-tab from the UK Data Service [here](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5340)
- Step 1: Create an account
- Step 2: Create a project and request access to the data
    - We use the `National Travel Survey, 2002-2023` dataset (SN: 5340)
- Step 3: Download TAB file format

`travel_times/`

- OPTIONAL Dataset - If it does not exist, it will be generated in the pipeline. They are added under oa/ or msoa/ subdirectories.
- e.g. oa/`travel_time_matrix.parquet` or msoa/`travel_time_matrix.parquet`

`ODWP01EW_OA.zip`
`ODWP15EW_MSOA_v1.zip`
`MSOA_2011_MSOA_2021_Lookup_for_England_and_Wales.csv`
`census_2011_rural_urban.csv`

```
