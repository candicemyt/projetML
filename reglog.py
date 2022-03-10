import numpy as np
from projet_etu import Loss
from projet_etu import Module

class MSELoss(Loss):
    def __init__(self):
        super().__init__()
    def forward(self, y, yhat):
        assert y.shape==yhat.shape
        res = np.linalg.norm(y[i]-yhat[i],2)**2
        assert len(res)==y.shape[0]
        return res

class Linear(Module):
    def __init__(self,input,output):
        super().__init__()
        self.input=input
        self.output=output
    def forward(self, X):
        ## Calcule la passe forward
        #calculer les sorties du module pour les entrées passées en paramètre
        assert X.shape[1]==len(self.input)
        res=[]
        m = MSELoss()
        assert res.shape[1]==len(self.out)
        return res


m=MSELoss()
y=np.ones((2,6))
yhat=np.ones((2,6))
yhat[1]=0
print(m.forward(y,yhat))