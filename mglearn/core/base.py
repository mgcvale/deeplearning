from abc import ABC, abstractmethod
import numpy as np
from mglearn.core.layer.base import Layer


class BaseModel(ABC):
    def __init__(self):
        self.layers: list[Layer] = []
        self.out: Layer = None

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass
