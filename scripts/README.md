# Scripts

## Synthetic Population Generation

- 1_prep_synthpop.py: Create a synthetic population using the SPC

## Adding Activity Patterns to Population

- 2_match_households_and_individuals.py: Match individuals in the synthetic population to travel diaries in the NTS. This is based on statistical matching approach described in ...

## Location Assignment

- 3.1_assign_primary_feasible_zones.py</ins>: This script is used to obtain, for each activity, the feasible destination zones that the activity could take place in. This is done by using a travel time matrix between zones to identify the zones that can be reached given the NTS reported travel time and travel mode in the NTS. A travel time matrix should be provided before running the pipeline (in the correct format). If a travel time matrix does not exist, the code can create travel time estimates based on mode average speeds and crow fly distance. For tips on creating a travel time matrix, see the comment here https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/20#issuecomment-2317037441
-  [3.2.1_assign_primary_zone_edu.py](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/main/scripts/3.2.1_assign_primary_zone_edu.py):
-  [3.2.2_assign_primary_zone_work.py](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/main/scripts/3.2.2_assign_primary_zone_work.py)
-  [3.2.3_assign_secondary_zone.py](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/main/scripts/3.2.3_assign_secondary_zone.py)
-  [3.3_assign_facility_all.py](https://github.com/Urban-Analytics-Technology-Platform/acbm/blob/main/scripts/3.3_assign_facility_all.py)

## Validation
- 4_validate.py: Validate the synthetic population by comparing the distribution of activity chains in the NTS to our model outputs.

## Output
