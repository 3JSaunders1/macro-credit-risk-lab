from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class IRFBundle:
    irfs: List[np.ndarray]     # list of (2x2)
    fevd: np.ndarray           # (H, 2, 2)
    model_type: str


class BaseTimeSeriesModel:
    """
    HARD CONTRACT:
    ALL MODELS MUST FOLLOW THIS EXACT INTERFACE
    """

    def fit(self, data):
        raise NotImplementedError

    def forecast(self, steps: int) -> np.ndarray:
        raise NotImplementedError

    def irf(self, horizon: int = 10) -> IRFBundle:
        raise NotImplementedError