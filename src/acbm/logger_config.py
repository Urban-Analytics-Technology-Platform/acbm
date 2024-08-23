import logging

import acbm
from acbm.utils import prepend_datetime

# Configure the root logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Shared console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Create a logger for the matching module
matching_logger = logging.getLogger("matching")
matching_file_handler = logging.FileHandler(
    acbm.logs_path / prepend_datetime("matching.log")
)
matching_file_handler.setLevel(logging.DEBUG)
matching_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
matching_logger.addHandler(matching_file_handler)
matching_logger.addHandler(console_handler)

# Create a logger for the primary assignment (feasible)
assigning_primary_feasible_logger = logging.getLogger("assigning_primary_feasible")
assigning_primary_feasible_handler = logging.FileHandler(
    acbm.logs_path / prepend_datetime("assigning_primary_feasible.log")
)
assigning_primary_feasible_handler.setLevel(logging.DEBUG)
assigning_primary_feasible_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
assigning_primary_feasible_logger.addHandler(assigning_primary_feasible_handler)
assigning_primary_feasible_logger.addHandler(console_handler)

# Create a logger for the primary assignment (locations)
assigning_primary_locations_logger = logging.getLogger("assigning_primary_locations")
assigning_primary_locations_handler = logging.FileHandler(
    acbm.logs_path / prepend_datetime("assigning_primary_locations.log")
)
assigning_primary_locations_handler.setLevel(logging.DEBUG)
assigning_primary_locations_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
assigning_primary_locations_logger.addHandler(assigning_primary_locations_handler)
assigning_primary_locations_logger.addHandler(console_handler)


# Create a logger for the secondary assignment
assigning_secondary_locations_logger = logging.getLogger(
    "assigning_secondary_locations"
)
assigning_secondary_locations_handler = logging.FileHandler(
    acbm.logs_path / prepend_datetime("assigning_secondary_locations.log")
)
assigning_secondary_locations_handler.setLevel(logging.DEBUG)
assigning_secondary_locations_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
assigning_secondary_locations_logger.addHandler(assigning_secondary_locations_handler)
assigning_secondary_locations_logger.addHandler(console_handler)
