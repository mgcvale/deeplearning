from abc import ABC, abstractmethod
from typing import List
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def backpropagate(self, y_true, y_pred) -> tuple[List[np.ndarray], List[np.ndarray]]:
        pass

    @abstractmethod
    def adjust(self, w_grad: List[np.ndarray], b_grad: List[np.ndarray]) -> List[np.ndarray]:
        pass