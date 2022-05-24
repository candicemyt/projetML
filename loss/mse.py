import numpy as np
from loss.loss import Loss


class MSELoss(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        """ Calcul du coût aux moindres carrés (mse).
            entrées : y -> batch*d
                    yhat -> batch*d
            sortie : res -> batch
        """
        assert y.shape == yhat.shape
        return np.linalg.norm((y-yhat),axis=1)**2

    def backward(self, y, yhat):
        """ Calcule le gradient du coût aux moindres carrés.
                entrées : y -> batch*d
                        yhat -> batch*d
                sortie : res -> batch*d
        """
        assert y.shape == yhat.shape
        return np.multiply(2,yhat-y)