import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from data.mltools import gen_arti, plot_data, plot_frontiere
from encapsulage import Sequentiel, mini_SGD
from loss.Mse import MSELoss
from modules.Linear import Linear

datax, datay = gen_arti(epsilon=0.3,data_type=0)
datay = datay.reshape(-1, 1)

lin1 = Linear(2, 1)
seq = Sequentiel([lin1])

n_iter = 100
ep = 1e-5
seq, loss = mini_SGD(seq, datax, datay, eps=ep, loss_fonction=MSELoss(),batch_size=100, nb_iteration=n_iter)

plt.figure()
plt.xlabel("nombre d'itération")
plt.ylabel('cout')
plt.plot(range(1,n_iter+1),loss)
plt.show()

def predict(datax):
    return np.where(seq.forward(datax)[-1]>0, 1, -1)

yhat=predict(datax)
print('taux de bonne classif : ',np.mean(yhat == datay))

plt.figure()
plt.title('Lineare')
plot_frontiere(datax,lambda x : predict(x),step=100)
plot_data(datax,datay.reshape(-1))
plt.show()

## Visualisation de la matrice de confusion
mat=confusion_matrix(datay, predict(datax))
print('matrice de confusion : ',mat)
plt.figure()
plt.title("matrice de confusion")
plt.imshow(mat)
plt.show()