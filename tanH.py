import numpy as np
from projet_etu import Module

class TanH(Module):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        #on met le gradient à 0
        #self.zero_grad()

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
                output : input*output
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