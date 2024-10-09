import logging

import acbm
from acbm.utils import prepend_datetime

# # Configure the root logger
# logging.basicConfig(
#     level=logging.WARNING,  # Set to DEBUG to capture all logs
#     format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
# )

# Shared console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set to WARNING for console output
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(filename)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
    )
)


def create_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(
        logging.DEBUG
    )  # Ensure the logger captures all messages at DEBUG level and above
    if not logger.hasHandlers():  # Check if the logger already has handlers
        file_handler = logging.FileHandler(acbm.logs_path / prepend_datetime(log_file))
        file_handler.setLevel(logging.DEBUG)  # Set to DEBUG for file output
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(filename)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # avoid logs from being propagated to the root logger (so that they don't show in the notebook)
        logger.propagate = False
    return logger


# Create loggers for different modules
preprocessing_logger = create_logger("preprocessing", "preprocessing.log")
matching_logger = create_logger("matching", "matching.log")
assigning_primary_feasible_logger = create_logger(
    "assigning_primary_feasible", "assigning_primary_feasible.log"
)
assigning_primary_zones_logger = create_logger(
    "assigning_primary_zone", "assigning_primary_zone.log"
)
assigning_secondary_zones_logger = create_logger(
    "assigning_secondary_zone", "assigning_secondary_zone.log"
)
assigning_facility_locations_logger = create_logger(
    "assigning_facility_locations", "assigning_facility_locations.log"
)

converting_to_matsim_logger = create_logger(
    "converting_to_matsim", "converting_to_matsim.log"
)
