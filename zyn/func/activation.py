from zyn.func.base import Activation
import numpy as np

class ReLU(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

    def derived(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return  1 / (1 + np.exp(-inputs))

    def derived(self, inputs: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-inputs))
        return out * (1 - out)

class Tanh(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)

    def derived(self, inputs: np.ndarray) -> np.ndarray:
        return 1 - self.forward(inputs)**2

