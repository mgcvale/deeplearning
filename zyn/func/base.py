from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derived(self, inputs: np.ndarray) -> np.ndarray:
        pass


class Loss(ABC):
    @abstractmethod
    def forward(self, y, y_pred):
        pass

    @abstractmethod
    def derived(self, y, y_pred):
        pass
