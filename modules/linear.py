import numpy as np
from modules.module import Module

class Linear(Module):

    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output
        # on initialise la matrice de poids (les poids sont compris entre -1 et 1)
        self._parameters = 2*(np.random.rand(input, output) - 0.5)
        # on met le gradient à 0
        self._gradient = np.zeros((self.input,self.output))

    def zero_grad(self):
        """Réinitialise à 0 le gradient"""
        self._gradient = np.zeros((self.input,self.output))

    def forward(self, X):
        """ Calcule la passe forward
            Calcule les sorties du module pour les entrées passées en paramètre
            entrée : X -> batch*input
            sortie : res -> batch*output
        """
        assert X.shape[1] == self.input
        return X @ self._parameters

    def backward_update_gradient(self, _input, delta):
        """ Met a jour le gradient
            Calcule le gradient du coût par rapport aux paramètres
            et l’additionne à la variable _gradient
            en fonction de l’entrée input et des δ de la couche suivante
                entrées : input -> batch*d
                        delta -> batch*output
                sortie : input*output
        """
        assert _input.shape[0] == delta.shape[0]
        assert _input.shape[1] == self.input
        assert delta.shape[1] == self.output

        self._gradient += _input.T @ delta

    def backward_delta(self, _input, delta):
        """ Calcule la derivee de l'erreur
            Calcule le gradient du coût par rapport aux entrées en fonction de l’entrée input
            et des deltas de la couche suivante
                entrées : input -> batch*input
                        delta -> batch*output
                sortie : res -> batch*input
        """
        assert _input.shape[1] == self._parameters.shape[0]
        assert delta.shape[1] == self._parameters.shape[1]
        assert _input.shape[0] == delta.shape[0]

        return delta @ (self._parameters.T)