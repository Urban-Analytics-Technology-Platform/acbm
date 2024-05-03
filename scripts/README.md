# Preparing synthetic population scripts

## Datasets
- [Synthetic Population Catalyst](https://github.com/alan-turing-institute/uatk-spc/blob/55-output-formats-python/python/README.md)
- [National Travel Survey](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5340)
- [Rural Urban Classification 2011 classification](https://geoportal.statistics.gov.uk/datasets/53360acabd1e4567bc4b8d35081b36ff/about)
- [OA centroids](): TODO

## Loading in the SPC synthetic population

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


## Matching
### Adding activity chains to synthetic populations
The purpose of this script is to match each individual in the synthetic population to a respondant from the [National Travel Survey (NTS)](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5340).

### Methods
We will try two methods:
   1. categorical matching: joining on relevant socio-demographic variables
   2.  statistical matching, as described in [An unconstrained statistical matching algorithm for combining individual and household level geo-specific census and survey data](https://doi.org/10.1016/j.compenvurbsys.2016.11.003).
