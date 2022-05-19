import numpy as np

from modules.Module import Module


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        if len(X.shape) > 1:
            # Matrice
            X = np.apply_along_axis(lambda x: np.exp(x - np.max(X)), 1, X)
            denom = np.apply_along_axis(lambda x: 1.0 / np.sum(x), 1, X)
            if len(denom.shape) == 1:
                denom = denom.reshape((denom.shape[0], 1))
            X = X * denom
        else:
            # Vecteur
            X = np.exp(X - np.max(X)).dot(1.0 / np.sum(np.exp(X - np.max(X))))
        return X

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass
