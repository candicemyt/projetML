import numpy as np
from module import Loss

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
        res = np.linalg.norm(y-yhat) ** 2
        return res

    def backward(self, y, yhat):
        """ Calcule le gradient du coût aux moindres carrés.
                entrées : y -> batch*d
                        yhat -> batch*d
                sortie : res -> batch*d
        """

        assert y.shape == yhat.shape
        return 2 * (y-yhat)

