from datetime import datetime
from typing import Any

import numpy as np
import tomlkit
from sklearn.metrics import mean_squared_error

import acbm


def get_config(config: str) -> dict[Any, Any]:
    with open(acbm.root_path / config, "rb") as f:
        return tomlkit.load(f)


def init_rng(config: dict):
    try:
        np.random.seed(config["parameters"]["seed"])
    except Exception as err:
        msg = f"config does not provide a rng seed with err: {err}"
        ValueError(msg)


def prepend_datetime(s: str, delimiter: str = "_") -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f"{current_date}{delimiter}{s}"


def calculate_rmse(predictions, targets):
    """
    Calculate the Root Mean Squared Error (RMSE) between predictions and targets,
    excluding NaN values.

    Parameters:
    - predictions (np.ndarray or pd.Series): The predicted values.
    - targets (np.ndarray or pd.Series): The actual values.

    Returns:
    - float: The RMSE value.
    """
    # Ensure inputs are numpy arrays
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Create a mask for non-NaN values
    mask = ~np.isnan(predictions) & ~np.isnan(targets)

    # Filter out NaN values
    filtered_predictions = predictions[mask]
    filtered_targets = targets[mask]

    # Calculate mean squared error
    mse = mean_squared_error(filtered_predictions, filtered_targets)

    # Calculate and return RMSE
    return np.sqrt(mse)
