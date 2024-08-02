"""
acbm: A package to create activity-based models (for transport demand modelling)
"""
from __future__ import annotations

import os
import pathlib

__all__ = ("__version__",)
__version__ = "0.1.0"

root_path = pathlib.Path(__file__).parent.parent.parent.resolve()
logs_path = root_path / "logs"
os.makedirs(logs_path, exist_ok=True)
