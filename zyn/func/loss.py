from zyn.func.base import Loss
import numpy as np

class MSE(Loss):
    def forward(self, y, y_pred):
        return 0.5 * ((y - y_pred) ** 2)

    def derived(self, y, y_pred):
        return y_pred - y

class BinaryCrossEntropy(Loss):
    def forward(self, y, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def derived(self, y, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y) / (y_pred * (1 - y_pred))

