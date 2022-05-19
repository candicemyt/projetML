import numpy as np

from modules.Module import Module


class Sigmoide(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        assert input.shape == delta.shape
        sig = self.forward(input)
        return np.multiply(np.multiply(sig, (1 - sig)), delta)
