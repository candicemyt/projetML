import numpy as np
from loss.loss import Loss

def Softmax(X):
    q = 1 / np.sum(np.exp(X), axis=1)
    p = np.exp(X)
    return p * q[:, None]

class SMCELoss(Loss):
    """calcul la loss cross entropique avec un Softmax passé au logarithme"""

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        tmp = Softmax(yhat)
        return -(tmp*y)+np.log(np.sum(np.exp(tmp)))

    def backward(self, y, yhat):
        return Softmax(yhat)-y
