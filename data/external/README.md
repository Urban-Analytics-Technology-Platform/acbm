The folder contains all external datasets necessary to run the pipeline. Some can be downloaded, while others need to be generated. The README.md file in this folder provides a guide on where to find / generate each dataset. In the future, we aim to:
- host some of the datasets on the cloud, or download them directly where possible in the pipeline
- add dataset paths in the config file

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
|   |   │   │       ├── OA
|   |   │   │       |   ├── travel_time_matrix.parquet
|   |   |   |       └── MSOA
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
            --output-formats parquet \
            --year 2020 \
            config/England/west-yorkshire.txt
        ```
        and replace `west-yorkshire` and `2020` with your preferred option.

`boundaries/`

- `MSOA_DEC_2021_EW_NC_v3.geojson`: This is the MSOA boundaries for England and Wales. It can be downloaded from [Data-Gov-UK](https://www.data.gov.uk/dataset/9dffb396-2934-43fb-9777-1aef704138ac/middle-layer-super-output-areas-december-2021-names-and-codes-in-ew-v3). If this link is no longer valid, the layer is also available from other sources
- `oa_england.geojson`: This is the OA boundaries for England (2021). The user can download it from [Data-Gov-UK](https://www.data.gov.uk/dataset/4d4e021d-fe98-4a0e-88e2-3ead84538537/output-areas-december-2021-boundaries-ew-bgc-v2)
- `study_area_zones.geojson`: This layer is the MSOAs / OAs in the study area. It is created in the pipeline (in [0_preprocess_inputs.py](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/main/scripts/0_preprocess_inputs.py)). The user does not have to worry about this file.

`centroids/`

- LSOA_Dec_2011_PWC_in_England_and_Wales_2022.csv
- `Output_Areas_Dec_2011_PWC_2022.csv`: An OA 2021 centroid layer. It can be downloaded from [Data-Gov-UK](https://www.data.gov.uk/dataset/ba661484-ceff-4a1c-91d8-3c57d0f0a933/output-areas-december-2011-ew-population-weighted-centroids_)

`nts/`

`UKDA-5340-tab`:
- Download the UKDA-5340-tab from the UK Data Service [here](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5340)
- Step 1: Create an account
- Step 2: Create a project and request access to the data
    - We use the `National Travel Survey, 2002-2023` dataset (SN: 5340)
- Step 3: Download TAB file format

`travel_times/`

- OPTIONAL Dataset - If it does not exist, it will be generated in the pipeline. They are added under oa/ or msoa/ subdirectories (e.g. oa/`travel_time_matrix.parquet` or msoa/`travel_time_matrix.parquet`). Columns are:
  - OA21CD_from / MSOA21CD_from: OA21CD code
  - OA21CD_to / MSOA21CD_to: OA21CD code
  - mode: ['pt', 'car', 'walk', 'cycle']
  - weekday: 1, 0
  - time_of_day: ['morning', 'afternoon', 'evening', 'night']
  - time: time in minutes

There is an [open issue](https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/20#issuecomment-2317037441) on denerating travel times which the user can use as a starting point IF they wish to generate the travel time matrix. In the future, we aim to add a script to generate the travel time matrix.

Other datasets (to be places in the root of the `external` folder):

- `ODWP01EW_OA.zip` & `ODWP15EW_MSOA_v1.zip`: These are commuting matrices from the census. They can be found in WICID data service. Go to [this link](https://wicid.ukdataservice.ac.uk/flowdata/cider/wicid/downloads.php), and search for `ODWP01EW_OA` and `ODWP15EW_MSOA` under 2021 Census
- `MSOA_2011_MSOA_2021_Lookup_for_England_and_Wales.csv`: This is the lookup table between MSOA 2011 and MSOA 2021. It can be downloaded from [Data-Gov-UK](https://www.data.gov.uk/dataset/da36cac8-51c4-4d68-a4a9-37ac47d2a4ba/msoa-2011-to-msoa-2021-to-local-authority-district-2022-exact-fit-lookup-for-ew-v2)
- `census_2011_rural_urban.csv`: OA level rural-urban classification. It can be downloaded from [ONS](https://geoportal.statistics.gov.uk/datasets/53360acabd1e4567bc4b8d35081b36ff/about). The classification is based on the 2011 Census, and the categories are: 'Urban major conurbation', 'Urban minor conurbation', 'Urban city and town', 'Urban city and town in a sparse setting': 'Rural town and fringe', 'Rural town and fringe in a sparse setting', 'Rural village', 'Rural village in a sparse setting', 'Rural hamlets and isolated dwellings', 'Rural hamlets and isolated dwellings in a sparse setting'

```
