import numpy as np
from modules.module import Module

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        e_x = np.exp(X)
        return e_x / e_x.sum(axis=1).reshape(-1, 1)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        s = self.forward(input)
        return np.multiply(np.multiply(s, (1 - s)), delta)