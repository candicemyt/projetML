import numpy as np

from loss.loss import Loss


class BCELoss(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        """ Calcul du coût cross entropique binaire.
            entrées : y -> batch*d
                    yhat -> batch*d
            sortie : res -> batch
        """
        assert y.shape == yhat.shape
        res = -np.sum(np.multiply(y,np.maximum(np.log(yhat), -100)) + np.multiply((1 - y),np.maximum(-100, np.log(1-yhat))), axis = 1)
        return res

    def backward(self, y, yhat):
        """ Calcule le gradient.
                entrées : y -> batch*d
                        yhat -> batch*d
                sortie : res -> batch*d
        """
        assert y.shape == yhat.shape
        return -y / (yhat) + ((1 - y) / (1 - yhat))
