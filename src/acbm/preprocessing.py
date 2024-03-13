import pandas as pd
import numpy as np



def nts_filter_by_year(data: pd.DataFrame, psu: pd.DataFrame, years: list) -> pd.DataFrame:
    '''
    Filter the NTS dataframe based on the chosen year(s)

    data: pandas DataFrame
        The NTS data to be filtered
    psu: pandas DataFrame
        The Primary Sampling Unit table in the NTS. It has the year 
    years: list
        The chosen year(s)
    '''
    # Check that all values of 'years' exist in the 'SurveyYear' column of 'psu'

    # Get the unique years in the 'SurveyYear' column of 'psu'
    unique_years = set(psu['SurveyYear'].unique())

    # Stop if any item in 'years' does not exist in the 'SurveyYear' column of 'psu'
    if not set(years).issubset(unique_years):
        # If not, print the years that do exist and stop execution
        print(f"At least one of the chosen year(s) do not exist in the PSU table. Years that exist in the PSU table are: {sorted(list(unique_years))}")
        return

    # Get the 'PSUID' values for the chosen year(s)
    psu_id_years = psu[psu['SurveyYear'].isin(years)]['PSUID'].unique()

    # Filter 'data' based on the chosen year
    data_years = data[data['PSUID'].isin(psu_id_years)]

    return data_years



def nts_filter_by_region(data: pd.DataFrame, psu: pd.DataFrame, regions: list) -> pd.DataFrame:
    '''
    Filter the NTS dataframe based on the chosen region(s)

    data: pandas DataFrame
        The NTS data to be filtered
    psu: pandas DataFrame
        The Primary Sampling Unit table in the NTS. It has the region assigned to each sample
    regions: list
        The chosen region(s)
    '''
    # 1. Create a column in the PSU table with the region names

    # Dictionary of the regions in the NTS and how they are coded
    region_dict = {
    -10.0: 'DEAD',
    -9.0: 'DNA',
    -8.0: 'NA',
    1.0: 'North East',
    2.0: 'North West',
    3.0: 'Yorkshire and the Humber',
    4.0: 'East Midlands',
    5.0: 'West Midlands',
    6.0: 'East of England',
    7.0: 'London',
    8.0: 'South East',
    9.0: 'South West',
    10.0: 'Wales',
    11.0: 'Scotland'
    }
    # In the PSU table, create a column with the region names
    psu["region_name"] = psu["PSUGOR_B02ID"].map(region_dict)

    # 2. Check that all values of 'years' exist in the 'SurveyYear' column of 'psu'

    # Get the unique regions in the 'PSUGOR_B02ID' column of 'psu'
    unique_regions = set(psu['region_name'].unique())
    # Stop if any item in 'regions' do not exist in the 'PSUGOR_B02ID' column of 'psu'
    if not set(regions).issubset(unique_regions):
        # If not, print the years that do exist and stop execution
        print(f"At least one of the chosen region(s) do not exist in the PSU table. Regions that exist in the PSU table are: {sorted(list(unique_regions))}")
        return
    
    # 3. Filter the 'data' df based on the chosen region(s)

    # Get the 'PSUID' values for the chosen year(s)
    psu_id_regions = psu[psu['region_name'].isin(regions)]['PSUID'].unique()
    # Filter 'data' based on the chosen year
    data_regions = data[data['PSUID'].isin(psu_id_regions)]

    return data_regions
