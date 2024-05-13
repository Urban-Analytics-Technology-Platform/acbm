# Activity-based Model

Guide to background and modelling assumptions made.

## Assumptions

### Geographies
Since the SPC currently uses 2011 OA11CD and MSOA11CD codes, 2011 boundaries will be used.

#### Locations
- Homes: OA11CD centroids
- Work: exact location data
- Education: exact location, travel time for a given mode is used to determine the feasible zone.
- Shop (food):
- Shop (other):
- Visit:

### Adding activity patterns to synthetic population

#### NTS data 
- We are currently using the entire NTS sample, but this could include trips with unrepresentative distances (e.g. commuting distance in London is not the same as liverpool). See https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/16

#### Household level matching 
- We use categorical matching at the household level (level 1) and then propensity score matching (PSM) at the individual level (level 2)
- We need to implement PSM from the beginning to ensure that each individual in the SPC is matched to at least one sample from the NTS. See https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/13
- Matching variables are decided using trial and error (see [2_match_households_and_individuals](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/d2f9e747c3d55148316661b13b1650fac4a5a4ad/notebooks/2_match_households_and_individuals.ipynb). Using PSM would allow us to use all variables
- For each SPC household, we randomly select one of the matched NTS households
- Rest of the assumptions are outlined in the [wiki page](https://github.com/Urban-Analytics-Technology-Platform/acbm/wiki/Adding-activity-patterns-to-synthetic-population)

#### Individual level matching 
- Done based on age_group and sex only. PSM without replacement
  
### Assigning activities to geographic locations

#### Mode and trip purpose mapping 

- The NTS has a detailed breakdown of modes. We create dictionaries to map these modes to more generic modes (less categories)
- The same is done for trip purposes
- You can see the mode_mapping and purp_mapping dictionaries in [3_locations_primary](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/3d1dc9bbd4651daa03c10c4e6140fc54cbdcc049/notebooks/3_locations_primary.ipynb)

#### Home locations 

- These are still MSOA centroids (as taken from SPC). These locations need to be edited so households are assigned to different houses

#### Activity locations (education)

##### Downloading and labeling POIs

- We download osm locations and label them. This is currently done through osmox with a custom config json file. See [this example](https://github.com/arup-group/osmox/blob/main/configs/config_UK.json)
- In the config file, we specify what to download from osm, and in "activity_mapping", we decide how each POI is labelled.
- For education POIs, I've done the following:   

> "kindergarden": ["education_kg", "work"],
> 
> "school": ["education_school", "work"],
> 
> "university": ["education_university", "work"],
> 
> "college": ["education_college", "work"],

##### Selecting feasible zones for each activity

- This is done in [get_possible_zones](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/c548fa7a6398dd0afde1398f7799e418b6068cd6/src/acbm/assigning.py#L201). The function looks at the travel mode and trip duration (from NTS), and uses a pre-calculated travel time matrix by mode to identify which zones can be reached for that mode and duration. It then selects activities that have a POI that matches our activity purpose `dact`
- The function can return an empty list for some activities, as no zones match our constraints. We need to add logic to ensure at least one zone with a relevant POI is chosen. We can choose a POI based on distance and disregard the travel time matrix

##### Selecting a zone from feasible zones

- If an individual in the NTS has an "education" activity, I map their age to an education type. See the age_group_mapping dictionary in 3_locations_primary:

> age_group_mapping = {
> 
> 1: "education_kg",   # "0-4"
> 
> 2: "education_school", # "5-10"
> 
> 3: "education_school", # "11-16"
> 
> 4: "education_university", # "17-20"
> 
> 5: "education_university", # "21-29"
> 
> 6: "education_university", # "30-39"
> 
> 7: "education_university", # "40-49"
> 
> 8: "education_university", # "50-59"
> 
> 9: "education_university" # "60+"
> }

    
- When selecting a location for an education activity in [select_zone](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/c548fa7a6398dd0afde1398f7799e418b6068cd6/src/acbm/assigning.py#L578), we try to select a zone that has a POI that matches the persons age group. If we can't we choose any other feasible zone with an education POI
- This logic should be moved upstream to the [get_possible_zone](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/c548fa7a6398dd0afde1398f7799e418b6068cd6/src/acbm/assigning.py#L201). For each activity, we should always ensure that our list of feasible zones has a zone with our specific POI category. This should be added in the [filter_by_activity](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/c548fa7a6398dd0afde1398f7799e418b6068cd6/src/acbm/assigning.py#L374) logic. The filter_by_activity logic currently looks at activity purpose from the NTS (e.g. "education"). We need to add the extra level of detail from age_group_mapping, and then filter based on that instead
- We select a zone from the feasible zones probabilistically based on total floor area of the POIs that match the relevant activity. See [select_zone](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/c548fa7a6398dd0afde1398f7799e418b6068cd6/src/acbm/assigning.py#L578)

##### Selecting a POI
- We choose a POI from the relevant POIs in the chosen zone.This is done probabilistically based on floor_area
- Should this step be embedded in `select_zone()`


#### TODO
- edit `osmox` config to replace education_college with education_university. We should have mutually exclusive groups only and these two options serve the same age group
- DONE [here](https://github.com/Urban-Analytics-Technology-Platform/acbm/commit/6acecb928ea2b9bf26952eb45b86f2918a6dccdf): migrate logic for age_group_mapping from `select_zone()` to `get_possible_zones()`
- edit `get_possible_zones()` to ensure it never returns an empty list of zones. See above for how to do this 

