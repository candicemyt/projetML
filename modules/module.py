class Module(object):
    def __init__(self):
        #stocke les paramètres du module lorsqu’il en a
        self._parameters = None
        #accumule le gradient calculé
        self._gradient = None

    def zero_grad(self):
        """permet de réinitialiser à 0 le gradient"""
        pass

    def forward(self, X):
        """ Calcule la passe forward
            Calcule les sorties du module pour les entrées passées en paramètre
        """
        pass

    def update_parameters(self, gradient_step=1e-3):
        """ met à jour les paramètres du module selon le gradient accumulé jusqu’à son appel
            avec un pas de gradient_step
        """

        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        """ Met a jour le gradient
            Calcule le gradient du coût par rapport aux paramètres
            et l’additionne à la variable _gradient
            en fonction de l’entrée input et des δ de la couche suivante
        """
        pass

    def backward_delta(self, input, delta):
        """ Calcule la derivee de l'erreur
            Calcule le gradient du coût par rapport aux entrées en fonction de l’entrée input
            et des deltas de la couche suivante
        """
        pass
