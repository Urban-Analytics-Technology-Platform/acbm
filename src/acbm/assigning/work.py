import logging
import random

import pandas as pd

# Define logger at the module level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a handler that outputs to the console
console_handler = logging.StreamHandler()
# Create a handler that outputs to a file
file_handler = logging.FileHandler("log_assigning.log")


# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Function to assign individuals while respecting the OD constraints
def select_work_zone_iterative(
    trips: dict, flow_constraints: dict, random_assignment: bool = False
) -> pd.DataFrame:
    """
    Assigns individuals to work zones while respecting the OD constraints from an external
    dataset (e.g. census flow data).

    This function iterates over each individual and their origin zones. For each individual,
    it creates a list of feasible destination zones that still have remaining flows. If there
    are such zones, it assigns the individual to one of them using a weighted random selection
    based on the remaining flows. If there are no feasible zones with remaining flows, the
    function either assigns the individual to a random feasible zone (if random_assignment is
    True) or skips the assignment (if random_assignment is False).

    After each assignment, the function updates the remaining flows between the origin and
    destination zones. The process continues until all individuals are assigned to a work zone .

    The function returns a DataFrame with the following columns:
    - 'activity_id': the ID of the individual activity
    - 'origin_zone': the origin zone of the individual activity
    - 'assigned_zone': the zone to which the individual activity was assigned
    - 'assignment_type': the method used to assign the individual activity ('Weighted' for weighted
    random selection based on remaining flows, 'Random' for random selection, or None if
    the assignment was skipped)

    Parameters
    ----------
    trips : dict
        A dictionary where the keys are the person/activity IDs and the values are dictionaries
        where keys are the origin zones and the values are the feasible destination zones.
        example: {164: {'E00059011': ['E00056917','E00056922', 'E00056923']},
                  165: {'E00059012': ['E00056918','E00056952', 'E00056923']}}

    flow_constraints : dict
        A dictionary where the keys are the origin-destination zone pairs and the values are the
        number of flows between the origin and destination zones. The intended use case is the
        UK census flow data.
        example: {('E00059011', 'E00056917'): 10,
                  ('E00059011', 'E00056922'): 5}

    random_assignment : bool, optional
        If True, the assignment of individuals to zones will be random when there are no feasible
        zones with remaining flows. If False, the assignment will be skipped for that individual.
        Default is False.
    """

    logger.info("Starting the assignment process.")
    assignments = []
    # Step 1: Initialize remaining flows dictionary directly from od_matrix
    remaining_flows = flow_constraints.copy()

    # Step 2: Iterate over each person and their origins
    for activity_id, origins in trips.items():
        for origin_id, feasible_zones in origins.items():
            logger.info(f"Processing activity {activity_id} from origin {origin_id}.")
            logger.debug(f"{activity_id}: {len(feasible_zones)} feasible zones")
            if feasible_zones:
                # Step 3: Create a list of feasible zones with their respective remaining flow weights
                weighted_zones = []
                for zone in feasible_zones:
                    flow = remaining_flows.get((origin_id, zone), 0)
                    if flow > 0:
                        weighted_zones.append((zone, flow))

                # Step 4: Perform weighted random OR random selection of a zone based on remaining flows
                logging.debug(
                    f"{activity_id}: {len(weighted_zones)} feasible zones with remaining flows"
                )
                if weighted_zones:
                    zones, weights = zip(*weighted_zones)
                    assigned_zone = random.choices(zones, weights=weights, k=1)[0]
                    assignment_type = "Weighted"
                    remaining_flows[(origin_id, assigned_zone)] -= 1
                    logger.info(
                        f"Assigned zone {assigned_zone} to person {activity_id} using weighted random selection."
                    )

                elif random_assignment:
                    # If there are no feasible zones with remaining flows and random_assignment is True,
                    #  choose a random feasible zone
                    assigned_zone = random.choice(feasible_zones)
                    assignment_type = "Random"
                    # remaining_flows[(origin_id, assigned_zone)] -= 1
                    logger.info(
                        f"Assigned zone {assigned_zone} to person {activity_id} using random selection."
                    )

                else:
                    # If there are no feasible zones with remaining flows and random_assignment is False, skip this person
                    assigned_zone = None
                    assignment_type = None
                    logger.info(
                        f"{activity_id}: No feasible zones with remaining flows for person. Assigned NA"
                    )

            # if there are no feasible zones
            else:
                logger.info(f"{activity_id}: No feasible zones. Assigned NA")
                assigned_zone = None
                assignment_type = None

            assignments.append(
                {
                    "activity_id": activity_id,
                    "origin_zone": origin_id,
                    "assigned_zone": assigned_zone,
                    "assignment_type": assignment_type,
                }
            )

    logger.info("Assignment process completed.")
    # Return the assignments as a DataFrame
    return pd.DataFrame(assignments)
