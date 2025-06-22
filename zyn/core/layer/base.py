from abc import ABC, abstractmethod
import numpy as np

from zyn.func.base import Activation


class Layer(ABC):
    def __init__(self, activation: Activation):
        self.weights: np.ndarray | None = None
        self.biases: np.ndarray | None = None
        self.activation = activation
        self.last_z: np.ndarray | None = None
        self.last_a: np.ndarray | None = None
        self.last_input: np.ndarray | None = None

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

