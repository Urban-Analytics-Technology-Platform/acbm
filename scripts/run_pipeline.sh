#/bin/bash

set -e

python scripts/0_preprocess_inputs.py --config_file $1
python scripts/0.1_run_osmox.py --config_file $1
python scripts/1_prep_synthpop.py --config_file $1
python scripts/2_match_households_and_individuals.py --config_file $1
python scripts/3.1_assign_primary_feasible_zones.py --config_file $1
python scripts/3.2.1_assign_primary_zone_edu.py --config_file $1
python scripts/3.2.2_assign_primary_zone_work.py --config_file $1
python scripts/3.2.3_assign_secondary_zone.py --config_file $1
python scripts/3.3_assign_facility_all.py --config_file $1
python scripts/4_validation.py --config_file $1
