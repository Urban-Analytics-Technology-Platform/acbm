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
    acbm.log_path / prepend_datetime("matching.log")
)
matching_file_handler.setLevel(logging.DEBUG)
matching_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
matching_logger.addHandler(matching_file_handler)
matching_logger.addHandler(console_handler)

# Create a logger for the primary assignment
assigning_primary_logger = logging.getLogger("assigning_primary")
assigning_primary_handler = logging.FileHandler(
    acbm.log_path / prepend_datetime("assigning_primary.log")
)
assigning_primary_handler.setLevel(logging.DEBUG)
assigning_primary_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
assigning_primary_logger.addHandler(assigning_primary_handler)
assigning_primary_logger.addHandler(console_handler)


# Create a logger for the secondary assignment
assigning_secondary_logger = logging.getLogger("assigning_secondary")
assigning_secondary_handler = logging.FileHandler(
    acbm.log_path / prepend_datetime("assigning_secondary.log")
)
assigning_secondary_handler.setLevel(logging.DEBUG)
assigning_secondary_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
assigning_secondary_logger.addHandler(assigning_secondary_handler)
assigning_secondary_logger.addHandler(console_handler)
