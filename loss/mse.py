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

        assert y.shape[0] == yhat.shape[0]
        return np.linalg.norm(y-yhat) ** 2

    def backward(self, y, yhat):
        """ Calcule le gradient du coût aux moindres carrés.
                entrées : y -> batch*d
                        yhat -> batch*d
                sortie : res -> batch*d
        """

        assert y.shape == yhat.shape
        return 2 * (yhat-y)

