import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from data.USPS_tools import load_usps
from encapsulage import Sequentiel, mini_SGD
from loss.smce import SoftMax_CELoss
from modules.linear import Linear
from modules.tanH import TanH

uspsdatatrain = "../data/USPS_train.txt"
uspsdatatest = "../data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain)
alltestx,alltesty = load_usps(uspsdatatest)
alltrainy = OneHotEncoder(sparse = False).fit_transform(alltrainy.reshape(-1,1))

alltrainx = MinMaxScaler().fit_transform(alltrainx)
alltestx = MinMaxScaler().fit_transform(alltestx)
lin1 = Linear(256, 100)
lin2 = Linear(100, 10)

seq = Sequentiel([lin1,TanH(),lin2,TanH()])

n_iter = 1000
eps = 1e-4

seq, loss = mini_SGD(seq,alltrainx, alltrainy, batch_size=100, eps=eps, loss_fonction = SoftMax_CELoss(), nb_iteration=n_iter)

outputs = seq.forward(alltestx)

yhat = np.argmax(outputs[-1], axis=1)

print('taux de bonne classif : ',np.mean(yhat == alltesty))

plt.figure()
plt.xlabel("nombre d'itération")
plt.ylabel('cout')
plt.plot(range(1, n_iter+1),loss)
plt.savefig('../out/loss_multiclass.png')
plt.show()

## Visualisation de la matrice de confusion
mat=confusion_matrix(alltesty, yhat)
print('matrice de confusion : ',mat)
plt.figure()
plt.title("matrice de confusion")
plt.imshow(mat)
plt.savefig('../out/matconf_multiclass.png')
plt.show()

idx = np.random.choice(len(alltestx), 4)
k=1
for i in idx:
    plt.figure()
    plt.title('y: '+str(alltesty[i])+' yhat :'+str(yhat[i]))
    plt.imshow(alltestx[i].reshape(16,16))
    plt.show()
    k += 1
