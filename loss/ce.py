import numpy as np
from loss.loss import Loss

def SoftMax(x):
    q=1/np.sum(np.exp(x), axis=1)
    p=np.exp(x)
    return p*q[:,None]

class SMCELoss(Loss):
    """calcul la loss cross entropique avec un Softmax passé au logarithme"""
    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        for yi in y:

        return

    def backward(self, y, yhat):
        #faire le onehot ici
