from zyn.core.layer.base import Layer
from zyn.func.activation import ReLU, Sigmoid, Tanh
from zyn.func.base import Activation
import numpy as np

class Dense(Layer):
    def __init__(self, in_shape: int, out_shape: int, activation: Activation):
        super().__init__(activation)
        self.in_shape = in_shape
        self.out_shape = out_shape

        if isinstance(activation, ReLU):
            self.weights = np.random.randn(out_shape, in_shape) * np.sqrt(2. / in_shape)
        elif isinstance(activation, (Sigmoid, Tanh)):
            self.weights = np.random.randn(out_shape, in_shape) * np.sqrt(1. / in_shape)
        else:
            self.weights = np.random.randn(out_shape, in_shape)

        self.biases = np.zeros(out_shape)  # bias are neuron-wise, so there is only an out_shape amount of them

    def forward(self, in_vals: np.ndarray) -> np.ndarray:
        self.last_input = in_vals
        self.last_z = np.dot(self.weights,
                             in_vals) + self.biases  # element-wise multiplication of every weight with every in_val to every out neuron
        self.last_a = self.activation.forward(self.last_z)
        return self.last_a

