from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tomlkit
from pydantic import BaseModel, Field
from sklearn.metrics import mean_squared_error


@dataclass(frozen=True)
class Parameters(BaseModel):
    seed: int
    region: str
    number_of_households: int | None = None
    zone_id: str
    travel_times: bool
    max_zones: int


class Config(BaseModel):
    parameters: Parameters = Field(description="Config: parameters.")

    @property
    def seed(self) -> int:
        return self.parameters.seed

    @property
    def region(self) -> str:
        return self.parameters.region

    @property
    def zone_id(self) -> str:
        return self.parameters.zone_id

    @property
    def max_zones(self) -> int:
        return self.parameters.max_zones

    @classmethod
    def origin_zone_id(cls, zone_id: str) -> str:
        return zone_id + "_from"

    @classmethod
    def destination_zone_id(cls, zone_id: str) -> str:
        return zone_id + "_to"

    # TODO: consider moving to method in config
    def init_rng(self):
        try:
            np.random.seed(self.seed)
        except Exception as err:
            msg = f"config does not provide a rng seed with err: {err}"
            ValueError(msg)


def load_config(filepath: str | Path) -> Config:
    with open(filepath, "rb") as f:
        return Config.model_validate(tomlkit.load(f))


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
