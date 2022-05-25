import numpy as np

from modules.module import Module

class ReLu(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return np.maximum(X, 0)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        assert input.shape == delta.shape
        return (np.where(input > 0, 1, 0)) * delta