import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# https://statistics.ukdataservice.ac.uk/dataset/england-and-wales-census-2021-ts058-distance-travelled-to-work
# df = pl.read_excel("data/external/TS058-Distance-Travelled-To-Work-2021-ltla-ONS.xlsx")
df = pl.read_excel("data/external/TS058-Distance-Travelled-To-Work-2021-msoa-ONS.xlsx")
print(df.columns)
fig, ax = plt.subplots(1, 1, figsize=(5, 4), squeeze=True)

for label, values in [
    (
        "Inc. WFH",
        [
            "Works mainly at an offshore installation, in no fixed place, or outside the UK",
            "Does not apply",
        ],
    ),
    (
        "Exc.WFH",
        [
            "Works mainly at an offshore installation, in no fixed place, or outside the UK",
            "Does not apply",
            "Works mainly from home",
        ],
    ),
]:
    # area_col = "Lower Tier Local Authorities"
    area_col = "Middle Layer Super Output Areas"
    fraction_travel = (
        df.join(
            df.group_by([area_col]).agg(pl.col("Observation").sum().alias("sum")),
            on=[area_col],
            how="left",
            coalesce=True,
        )
        .select(
            [
                pl.col(area_col),
                pl.col("Distance travelled to work (11 categories)"),
                pl.col("Observation"),
                (pl.col("Observation") / pl.col("sum")).alias("frac"),
            ]
        )
        .filter(~pl.col("Distance travelled to work (11 categories)").is_in(values))
        .group_by(area_col)
        .agg(pl.col("frac").sum())
    )
    fraction_travel.get_column("frac").alias(label).to_pandas().plot(
        kind="hist", bins=np.arange(0, 1, 0.01), ax=ax
    )
    ax.axvline(
        fraction_travel.select("frac").median().to_numpy()[0, 0], c="grey", ls=":"
    )
plt.legend()
plt.show()
