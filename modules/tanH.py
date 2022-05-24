import numpy as np
from modules.module import Module


class TanH(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        assert input.shape == delta.shape
        return np.multiply((1 - np.square(self.forward(input))),delta)
