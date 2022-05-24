import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from data.mltools import gen_arti, plot_data, plot_frontiere
from encapsulage import Sequentiel, mini_SGD
from loss.mse import MSELoss
from modules.linear import Linear

datax, datay = gen_arti(epsilon=0.5,data_type=0,nbex=3000)
datay = datay.reshape(-1, 1)

datax_train, datax_test, datay_train, datay_test = train_test_split(datax, datay)

lin1 = Linear(2, 1)
seq = Sequentiel([lin1])

n_iter = 100
ep = 1e-5
seq, loss = mini_SGD(seq, datax_train, datay_train, eps=ep, loss_fonction=MSELoss(),batch_size=100, nb_iteration=n_iter)

plt.figure()
plt.xlabel("nombre d'itération")
plt.ylabel('cout')
plt.plot(range(1,n_iter+1),loss)
plt.savefig(f"../out/loss_linear.png")
plt.show()

def predict(datax):
    return np.where(seq.forward(datax)[-1]>0, 1, -1)

yhat_train=predict(datax_train)
print('taux de bonne classif en train : ',np.mean(yhat_train == datay_train))

yhat_test=predict(datax_test)
print('taux de bonne classif en test : ',np.mean(yhat_test == datay_test))

plt.figure()
plot_frontiere(datax_test,lambda x : predict(x),step=100)
plot_data(datax_test,datay_test.reshape(-1))
plt.savefig(f"../out/dataviz_linear.png")
plt.show()

## Visualisation de la matrice de confusion
mat=confusion_matrix(datay_test, predict(datax_test))
print('matrice de confusion : ',mat)
plt.figure()
plt.title("matrice de confusion")
plt.imshow(mat)
plt.show()