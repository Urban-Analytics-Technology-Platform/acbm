import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tomlkit
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Parameters(BaseModel):
    seed: int
    region: str
    number_of_households: int | None = None
    zone_id: str
    travel_times: bool
    boundary_geography: str


@dataclass(frozen=True)
class WorkAssignmentParams(BaseModel):
    use_percentages: bool
    weight_max_dev: float
    weight_total_dev: float
    max_zones: int


class Config(BaseModel):
    parameters: Parameters = Field(description="Config: parameters.")
    work_assignment: WorkAssignmentParams = Field(
        description="Config: parameters for work assignment."
    )

    @property
    def seed(self) -> int:
        return self.parameters.seed

    @property
    def region(self) -> str:
        return self.parameters.region

    @property
    def zone_id(self) -> str:
        return self.parameters.zone_id

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
            random.seed(self.seed)
        except Exception as err:
            msg = f"config does not provide a rng seed with err: {err}"
            ValueError(msg)


def load_config(filepath: str | Path) -> Config:
    with open(filepath, "rb") as f:
        return Config.model_validate(tomlkit.load(f))
