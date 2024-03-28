import numpy as np
import pandas as pd


def nts_filter_by_year(
    data: pd.DataFrame, psu: pd.DataFrame, years: list
) -> pd.DataFrame:
    """
    Filter the NTS dataframe based on the chosen year(s)

    data: pandas DataFrame
        The NTS data to be filtered
    psu: pandas DataFrame
        The Primary Sampling Unit table in the NTS. It has the year
    years: list
        The chosen year(s)
    """
    # Check that all values of 'years' exist in the 'SurveyYear' column of 'psu'

    # Get the unique years in the 'SurveyYear' column of 'psu'
    unique_years = set(psu["SurveyYear"].unique())

    # Stop if any item in 'years' does not exist in the 'SurveyYear' column of 'psu'
    if not set(years).issubset(unique_years):
        # If not, print the years that do exist and stop execution
        print(
            f"At least one of the chosen year(s) do not exist in the PSU table. Years that exist in the PSU table are: {sorted(list(unique_years))}"
        )
        return None

    # Get the 'PSUID' values for the chosen year(s)
    psu_id_years = psu[psu["SurveyYear"].isin(years)]["PSUID"].unique()

    # Filter 'data' based on the chosen year
    data_years = data[data["PSUID"].isin(psu_id_years)]

    return data_years


def nts_filter_by_region(
    data: pd.DataFrame, psu: pd.DataFrame, regions: list
) -> pd.DataFrame:
    """
    Filter the NTS dataframe based on the chosen region(s)

    data: pandas DataFrame
        The NTS data to be filtered
    psu: pandas DataFrame
        The Primary Sampling Unit table in the NTS. It has the region assigned to each sample
    regions: list
        The chosen region(s)
    """
    # 1. Create a column in the PSU table with the region names

    # Dictionary of the regions in the NTS and how they are coded
    region_dict = {
        -10.0: "DEAD",
        -9.0: "DNA",
        -8.0: "NA",
        1.0: "North East",
        2.0: "North West",
        3.0: "Yorkshire and the Humber",
        4.0: "East Midlands",
        5.0: "West Midlands",
        6.0: "East of England",
        7.0: "London",
        8.0: "South East",
        9.0: "South West",
        10.0: "Wales",
        11.0: "Scotland",
    }
    # In the PSU table, create a column with the region names
    psu["region_name"] = psu["PSUGOR_B02ID"].map(region_dict)

    # 2. Check that all values of 'years' exist in the 'SurveyYear' column of 'psu'

    # Get the unique regions in the 'PSUGOR_B02ID' column of 'psu'
    unique_regions = set(psu["region_name"].unique())
    # Stop if any item in 'regions' do not exist in the 'PSUGOR_B02ID' column of 'psu'
    if not set(regions).issubset(unique_regions):
        # If not, print the years that do exist and stop execution
        print(
            f"At least one of the chosen region(s) do not exist in the PSU table. Regions that exist in the PSU table are: {sorted(list(unique_regions))}"
        )
        return None

    # 3. Filter the 'data' df based on the chosen region(s)

    # Get the 'PSUID' values for the chosen year(s)
    psu_id_regions = psu[psu["region_name"].isin(regions)]["PSUID"].unique()
    # Filter 'data' based on the chosen year
    data_regions = data[data["PSUID"].isin(psu_id_regions)]

    return data_regions


def transform_by_group(
    data: pd.DataFrame,
    group_col: str,
    transform_col: str,
    new_col: str,
    transformation_type: str,
) -> pd.DataFrame:
    """
    Group the dataframe by the 'group_col' and apply the 'transformation_type' to the 'transform_col' for each group

    data: pandas DataFrame
        The data to be grouped and transformed
    group_col: str
        The column to group by
    transform_col: str
        The column to transform
    new_col: str
        The new column to store the result
    transformation_type: str
        The type of transformation ('sum', 'mean', 'max', 'min', etc.)
    """
    # create a copy of the data df
    data_copy = data.copy()
    # check that the 'transform_col' is a numeric column. If not, try to transfrom it to numeric
    if data_copy[transform_col].dtype not in [np.float64, np.int64]:
        try:
            data_copy[transform_col] = pd.to_numeric(data_copy[transform_col])
        # if transformation fails, return the original data_copy
        except:
            print(f"The column '{transform_col}' could not be transformed to numeric")
            return data_copy
    # Group the data by 'group_col' and apply the 'transformation_type' to the 'transform_col' for each group.
    # The result is stored in a new column called 'new_col'
    data_copy[new_col] = data_copy.groupby(group_col)[transform_col].transform(
        transformation_type
    )

    return data_copy


def num_adult_child_hh(
    data: pd.DataFrame, group_col: str, age_col: str
) -> pd.DataFrame:
    """Calculates the number of adults and children in each household.

    Parameters
    ----------
    data: pandas DataFrame
        The dataframe to be used
    group_col: str
        The column to group by
    age_col: str
        column with age of each individual

    Returns
    -------
    data: pandas DataFrame
        The original dataframe with these new columns: is'adult', 'num_adults', 'is_child', 'num_children', 'is_pension_age', 'num_pension_age'
    """
    data = data.assign(
        is_adult=(data[age_col] >= 16).astype(int),
        num_adults=lambda df: df.groupby(group_col)["is_adult"].transform("sum"),
        is_child=(data[age_col] < 16).astype(int),
        num_children=lambda df: df.groupby(group_col)["is_child"].transform("sum"),
        is_pension_age=(data[age_col] >= 65).astype(int),
        num_pension_age=lambda df: df.groupby(group_col)["is_pension_age"].transform(
            "sum"
        ),
    )

    return data


def count_per_group(
    df: pd.DataFrame, group_col: str, count_col: str, values: list, value_names: list
) -> pd.DataFrame:
    """
    Group the DataFrame by 'group_col', count the number of occurrences of
    specific values in 'count_col' within each group, and return a new DataFrame
    with a column for each value specified.

    Parameters:
    df: pandas DataFrame
        The DataFrame to group and count values in.
    group_col: str
        The name of the column to group by.
    count_col: str
        The name of the column to count values in.
    values: list
        The values to count. e.g. [1, 5, 7]
    value_names: list
        The names to use for the new columns in the output. e.g. ['col1', 'col2']

    Returns:
    DataFrame: A pandas DataFrame where the index is the group labels and
               the columns are the value_names. The values are the counts of
               each value within each group.
    """
    # Group the DataFrame by 'group_col' and count the occurrences of each value in 'count_col'
    counts = df.groupby(group_col)[count_col].value_counts()

    # Initialize an empty DataFrame to store the results
    result = pd.DataFrame()

    # We only want to report specific values. For each value to report (count), create a new column in the result DataFrame
    for val, name in zip(values, value_names):
        result[name] = (
            counts.xs(val, level=count_col)
            .reindex(df[group_col].unique())
            .fillna(0)
            .astype(int)
        )  # reindex so as not to drop groups that don't have the specified values

    return result


def truncate_values(x: int, lower: int = None, upper: int = None) -> int:
    """
    Limit the value of x to the range [lower, upper]

    Parameters
    ----------
    x: int
        The value to be limited
    lower: int
        The lower bound of the range
    upper: int
        The upper bound of the range

    Returns
    -------
    int
        The value of x, limited to the range [lower, upper]
    """
    if upper is not None:
        if x > upper:
            return upper
    if lower is not None:
        if x < lower:
            return lower
    return x
