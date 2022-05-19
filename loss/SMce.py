import numpy as np

from loss.Loss import Loss
from modules.SoftMax import Softmax


class SoftMax_CELoss(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        """ Calcul du coût cross entropique avec un softMax passé au logarithme.
            entrées : y -> batch*d
                    yhat -> batch*d
            sortie : res -> batch
        """
        assert y.shape == yhat.shape
        yhat = Softmax().forward(yhat)
        return np.sum(-(y * np.log(yhat)), axis=1)

    def backward(self, y, yhat):
        """ Calcule le gradient.
                entrées : y -> batch*d
                        yhat -> batch*d
                sortie : res -> batch*d
        """
        assert y.shape == yhat.shape
        nz = np.nonzero(y)
        yhat[nz] -= 1
        return yhat