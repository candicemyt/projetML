import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, rand_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from data.USPS_tools import load_usps
from encapsulage import Sequentiel, mini_SGD
from loss.bce import BCELoss
from modules.linear import Linear
from modules.sigmoide import Sigmoide
from modules.tanH import TanH

uspsdatatrain = "../../../data/USPS_train.txt"
uspsdatatest = "../../../data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain)
alltestx,alltesty = load_usps(uspsdatatest)
alltrainy = OneHotEncoder(sparse = False).fit_transform(alltrainy.reshape(-1,1))

alltrainx = MinMaxScaler().fit_transform(alltrainx)
alltestx = MinMaxScaler().fit_transform(alltestx)
lin1 = Linear(256, 50)
lin2 = Linear(50, 5)
lin3 = Linear(5, 50)
lin4 = Linear(50, 256)

seq = Sequentiel([lin1,TanH(),lin2,TanH(),lin3,TanH(),lin4,Sigmoide()])

n_iter = 1000
eps = 1e-4

seq, loss = mini_SGD(seq,alltrainx, alltrainx, batch_size=100, eps=eps, loss_fonction = BCELoss(), nb_iteration=n_iter)

#print(yhat)
idx = np.random.choice(len(alltestx), 5)
for i in idx:
    plt.figure()
    plt.title('image originale de '+str(alltesty[i]))
    plt.imshow(alltestx[i].reshape(16,16))
    plt.show()
    Xhat = seq.forward(alltestx)[-1]
    plt.figure()
    plt.title('image reconstruite de ' + str(alltesty[i]))
    plt.imshow(Xhat[i].reshape(16,16))
    plt.show()

plt.figure()
plt.xlabel("nombre d'itération")
plt.ylabel('cout')
plt.plot(range(1, n_iter+1),loss)
plt.show()

km = KMeans(n_clusters=10)
Xhat = seq.forward(alltrainx)[-1]
ypred = km.fit_predict(Xhat)
print('homogeneity train :', homogeneity_score(alltrainy,ypred))
print('rand_index train :', rand_score(alltrainy,ypred))

km = KMeans(n_clusters=10)
Xhat = seq.forward(alltestx)[-1]
ypred = km.fit_predict(Xhat)
print('homogeneity test :', homogeneity_score(alltesty,ypred))
print('rand_index test :', rand_score(alltesty,ypred))