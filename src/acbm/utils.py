from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import acbm
from acbm.config import Config


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


def get_travel_times(config: Config) -> pd.DataFrame:
    if config.parameters.travel_times:
        return pd.read_parquet(
            acbm.root_path / "data/external/travel_times/oa/travel_time_matrix.parquet"
        )
    return pd.read_parquet(
        acbm.root_path / config.interim_path / "travel_time_estimates.parquet"
    )
