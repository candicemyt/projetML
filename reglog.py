import numpy as np
from projet_etu import Loss
from projet_etu import Module

class MSELoss(Loss):
    def __init__(self):
        super().__init__()
    def forward(self, y, yhat):
        """ Calcul du coût aux moindres carrés (mse).
            input : y est de taille batch*d
                    yhat est de taille batch*d
            output : res est de taille batch
        """
        assert y.shape==yhat.shape
        res = np.linalg.norm( y-yhat, axis=1) ** 2
        assert len(res)==y.shape[0]
        return res
    def backward(self, y, yhat):
        """ Calcule le gradient du coût aux moindres carrés.
                input : y est de taille batch*d
                        yhat est de taille batch*d
                output : res est de taille batch*d
        """
        assert y.shape == yhat.shape
        res = 2 * (y-yhat)
        assert res.shape == y.shape
        return res

class Linear(Module):
    def __init__(self,input,output):
        super().__init__()
        self.input=input
        self.output=output
    def forward(self, X):
        """ Calcule la passe forward
                input : X est de taille batch*input
                output : res est de taille batch*output
        """
        assert X.shape[1]==len(self.input)
        assert res.shape[1]==len(self.out)
        assert X.shape[0]==res.shape[0]
        return res


m=MSELoss()
y=np.ones((2,6))
yhat=np.ones((2,6))
yhat[1]=0
print(m.forward(y,yhat))
print(m.backward(y,yhat))