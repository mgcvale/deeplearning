from typing import List

from zyn.core.base import BaseModel
from zyn.func.base import Loss
from zyn.optim.base import Optimizer

import numpy as np


class SGD(Optimizer):
    def __init__(self, model: BaseModel, loss: Loss, lr=0.01):
        self.lr = lr
        self.model = model
        self.loss = loss

    def backpropagate(self, y_true: float, y_pred: float) -> tuple[List[np.ndarray], List[np.ndarray]]:
        if self.model.out.last_z is None:
            raise RuntimeError("Cannot backpropagate before forward pass: no stored pre-activations")

        w_grads = []
        b_grads = []

        # first, we calculate the gradient from the loss w.r.t the y-pred
        # calculate ∂L/∂ŷ (how loss changes w.r.t to the predicted output)
        dL_dy_hat = self.loss.derived(y_true, y_pred)

        # calculate ∂ŷ/∂z (how the predicted output changes w.r.t to the pre-activation of the last layer
        dy_hat_dz = self.model.out.activation.derived(self.model.out.last_z)

        # calculate δ for output layer
        delta = dL_dy_hat * dy_hat_dz

        prev_layer = self.model.out
        prev_delta = delta

        # add the first gradients to the list
        w_grads.append(np.outer(delta, self.model.out.last_input))
        b_grads.append(delta)

        # -2 to exclude output, stop before -1 (0), step by -1 (go backward)
        for i in range(len(self.model.layers) - 2, -1, -1):
            layer = self.model.layers[i]

            # calculate delta for this layer
            df_dz = layer.activation.derived(layer.last_z)  # calculate f'(z) = ∂f/∂z - "undo" the activation function

            wt_delta = prev_layer.weights.T @ prev_delta  # W(n+1)T * delta(n+1)
            delta = wt_delta * df_dz  # chain rule -  δ(n) = W(n+1)T . δ(n+1) ⊙ f′(z(n))

            # calculate the weight and bias gradients with this delta
            b_grads.append(delta)
            w_grads.append(np.outer(delta, layer.last_input))

            prev_layer = layer
            prev_delta = delta

        return w_grads[::-1], b_grads[::-1]

    def adjust(self, w_grads, b_grads):

        # the gradients must match the model layer count
        layer_count = len(self.model.layers)
        if layer_count != len(w_grads):
            raise RuntimeError(
                f"Cannot adjust model; weight gradient amount doesn't match layer amount: {layer_count} layers; {len(w_grads)} weight gradients")
        if layer_count != len(b_grads):
            raise RuntimeError("Cannot adjust model; bias gradient amount doesn't match layer amount")

        for layer, layer_w_grads, layer_b_grads in zip(self.model.layers, w_grads, b_grads):
            # first, we update the weights
            layer.weights -= layer_w_grads * self.lr

            # now, the biases
            layer.biases -= layer_b_grads * self.lr

