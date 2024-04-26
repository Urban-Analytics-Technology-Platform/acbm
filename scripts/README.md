# Preparing synthetic population scripts

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
            config/England/west-yorkshire.txt --year 2020
        ```
        and replace `west-yorkshire` and `2020` with your preferred option.


## Matching
TODO
