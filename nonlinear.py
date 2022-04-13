import numpy as np
from module import Module

class TanH(Module):
    def __init__(self):
        super().__init__()


    def forward(self, X):
        print('X',X)
        return np.tanh(X)

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        return delta * (1 - np.square(input))



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
        #return delta * input * (1 - input)
        return np.multiply((1/(1+np.exp(-input))*(1-1/1+np.exp(-input))),delta)