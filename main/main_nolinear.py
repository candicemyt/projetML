import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from data.mltools import gen_arti, plot_frontiere, plot_data
from encapsulage import Sequentiel, mini_SGD
from loss.mse import MSELoss
from modules.linear import Linear
from modules.sigmoide import Sigmoide
from modules.tanH import TanH


def proj_biais(datax):
    return np.hstack((np.ones(datax.shape[0]).reshape(-1,1),datax))

datax, datay = gen_arti(epsilon=0.1,data_type=1)
datay = datay.reshape(-1,1)
datay_train = np.where(datay==-1,0,1)
datax_train = proj_biais(datax)
print(datax_train.shape)

datax, datay = gen_arti(epsilon=0.1,data_type=1)
datay = datay.reshape(-1,1)
datay_test = np.where(datay==-1,0,1)
datax_test = proj_biais(datax)

lin1 = Linear(3, 10)
lin2 = Linear(10, 1)
ae = Sequentiel([lin1,TanH(),lin2,Sigmoide()])
n_iter = 1000
ep = 1e-4
seq, loss = mini_SGD(ae, datax_train, datay_train, batch_size=1000, eps=ep, loss_fonction=MSELoss(), nb_iteration=n_iter)


def predict(datax):
    datax_b = proj_biais(datax)
    outputs=seq.forward(datax_b)
    return np.where(outputs[-1] > 0.5, 1, -1)

yhat_train=predict(datax_train)
print('taux de bonne classif train: ',np.mean(yhat_train == datay_train))

yhat_test=predict(datax_test)
print('taux de bonne classif test : ',np.mean(yhat_test == datay_test))

plt.figure()
plt.plot(range(1,n_iter+1),loss)
plt.xlabel("nombre d'itération")
plt.ylabel('cout')
plt.savefig('../out/loss_nonlinear.png')
plt.show()

plt.figure()
plt.title('No linear')
plot_frontiere(datax_test,lambda x : predict(x),step=100)
plot_data(datax_test,datay_test.reshape(-1))
plt.savefig('../out/dataviz_nonlinear.png')
plt.show()

## Visualisation de la matrice de confusion
mat=confusion_matrix(datay_test, predict(datax_test))
print('matrice de confusion : ',mat)
plt.figure()
plt.title("matrice de confusion")
plt.imshow(mat)
plt.show()