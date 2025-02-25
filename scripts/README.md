# Scripts

## Synthetic Population Generation

- `1_prep_synthpop.py`: Create a synthetic population using the SPC

## Adding Activity Patterns to Population

- `2_match_households_and_individuals.py`: Match individuals in the synthetic population to travel diaries in the NTS. This is based on categorical matching (household levwel) followed by statistical matching (individual level) within housholds

## Location Assignment

- `3.1_assign_primary_feasible_zones.py`: This script is used to obtain, for each activity, the feasible destination zones that the activity could take place in. This is done by using a travel time matrix between zones to identify the zones that can be reached given the NTS reported travel time and travel mode in the NTS. A travel time matrix should be provided before running the pipeline (in the correct format). If a travel time matrix does not exist, the code can create travel time estimates based on mode average speeds and crow fly distance. For tips on creating a travel time matrix, see the comment here https://github.com/Urban-Analytics-Technology-Platform/acbm/issues/20#issuecomment-2317037441
- `3.2.1_assign_primary_zone_edu.py`: Assign individuals education activiites to zones based on the reported distance, a specified tolerance parameter (distance +- tolerance), and a detour factor (that relates euclidian distance to network distance)
- `3.2.2_assign_primary_zone_work.py`: Assigns individual work activities to zones using an optimization problem that attempts to minimize deviation between assigned and reported distances
- `3.2.3_assign_secondary_zone.py`: Assign individual secondary activities to zones using a spacetime approach, with primary activity zones used as anchors. This uses the solver in the opns source package PAM
- `3.3_assign_facility_all.py`: Assign activities to specific facilities with point locations. For each activity, facilities of the relevant type inside the zone assigned to the activity are sampled. Sampling can be random or based on floor area

## Validation
- `4_validate.py`: Validate the synthetic population by comparing the distribution of activity chains in the NTS to our model outputs. The output is a number of plots that look at self-consistency

## Postprocessing

- `5_acbm_to_matsim_xml.py`: Convert dataframe outputs of pipeline to an xml that can be used as input to MATSim. This also includes adding some individual attributes (e.g whether the person is a student)
