import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        #stocke les paramètres du module lorsqu’il en a
        self._parameters = None
        #accumuler le gradient calculé
        self._gradient = None

    def zero_grad(self):
        ## permet de réinitialiser à 0 le gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        #calculer les sorties du module pour les entrées passées en paramètre
        # appeler successivement les fonctions forward de chaque module
        # avec comme entrée la sortie du précédent
        pass

    def update_parameters(self, gradient_step=1e-3):
        #met à jour les paramètres du module selon le gradient accumulé jusqu’à son appel
        #avec un pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        #calculer le gradient du coût par rapport aux paramètres
        #et l’additionner à la variable _gradient
        #en fonction de l’entrée input et des δ de la couche suivante delta
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        #calculer le gradient du coût par rapport aux entrées en fonction de l’entrée input
        #et des deltas de la couche suivante delta
        # le dernier module calcule le gradient par rapport à ses paramètres
        # et les deltas qu’il doit rétro-propager (à partir des deltas du loss)
        # puis en parcourant en sens inverse le réseau, chaque module répète la même opération :
        # le calcul de la mise à jour de son gradient (backward_update_gradient)
        # et le delta qu’il doit transmettre à la couche précédente (backward_delta).
        pass
