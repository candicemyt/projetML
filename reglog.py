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
        assert y.shape == yhat.shape
        res = np.linalg.norm(y-yhat, axis=1) ** 2
        assert len(res) == y.shape[0]
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
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        #on initialise la matrice de poids
        self._parameters = 2 * (np.random.rand(self.input, self.output) - 0.5)
        #on met le gradient à 0
        self.zero_grad()
    def zero_grad(self):
        """permet de réinitialiser à 0 le gradient"""
        self._gradient = np.zeros((self.input, self.output))
    def forward(self, X):
        """ Calcule la passe forward
            Calcule les sorties du module pour les entrées passées en paramètre
                input : X est de taille batch*input
                output : res est de taille batch*output
        """
        assert X.shape[1] == self.input
        res = X @ self._parameters
        assert res.shape[1] == self.output
        assert X.shape[0] == res.shape[0]
        return res
    def backward_update_gradient(self, input, delta):
        """ Met a jour la valeur du gradient
            Calcule le gradient du coût par rapport aux paramètres
            et l’additionne à la variable _gradient
            en fonction de l’entrée input et des δ de la couche suivante delta
                input : input est de taille batch*input
                        delta est de taille batch*output
        """
        assert input.shape[0] == delta.shape[0]
        assert input.shape[1] == self.input
        assert delta.shape[1] == self.output
        self._gradient += input.T @ delta
    def backward_delta(self, input, delta):
        """ Calcule la derivee de l'erreur
            Calcule le gradient du coût par rapport aux entrées en fonction de l’entrée input
            et des deltas de la couche suivante delta
                input : input est de taille batch*input
                        delta est de taille batch*output
                output : res est de taille batch*input
        """
        assert input.shape[0] == delta.shape[0]
        assert input.shape[1] == self.input
        assert delta.shape[1] == self.output
        res = delta @ (self._parameters.T)
        assert res.shape[0] == input.shape[0]
        assert res.shape[1] == self.input
        return res

#test de l'implémentation en réalisant une boucle d'apprentissage par descente de gradient
datax = np.random.randn(10,10)
print(datax)
datay = np.random.choice([-1,1],10)
print(datay)
mse = MSELoss()
linear = Linear(10,1)
res_lin = linear.forward(datax)
res_mse = mse.forward(datay.reshape(-1,1), res_lin)
delta_mse = mse.backward(datay.reshape(-1,1),res_lin)
linear.backward_update_gradient(datax,delta_mse)
grad_lin = linear._gradient
delta_lin = linear.backward_delta(datax,delta_mse)