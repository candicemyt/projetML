import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from data.USPS_tools import load_usps
from encapsulage import Sequentiel, mini_SGD
from loss.bce import BCELoss
from modules.linear import Linear
from modules.sigmoide import Sigmoide
from modules.tanH import TanH

uspsdatatrain = "../../data/USPS_train.txt"
uspsdatatest = "../../data/USPS_test.txt"
alltrainx,alltrainy = load_usps(uspsdatatrain)
alltestx,alltesty = load_usps(uspsdatatest)

alltrainx = MinMaxScaler().fit_transform(alltrainx)
alltestx = MinMaxScaler().fit_transform(alltestx)

alltrainx_noise = alltrainx + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=alltrainx.shape)
alltestx_noise = alltestx + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=alltestx.shape)

alltrainx_noise = np.clip(alltrainx_noise, 0., 1.)
alltestx_noise = np.clip(alltestx_noise, 0., 1.)

lin1 = Linear(256, 100)
lin2 = Linear(100, 10)
lin3 = Linear(10, 100)
lin4 = Linear(100, 256)

seq = Sequentiel([lin1,TanH(),lin2,TanH(),lin3,TanH(),lin4,Sigmoide()])

n_iter = 1000
ep = 1e-3
sq, loss = mini_SGD(seq,alltrainx_noise, alltrainx, batch_size=100,eps=ep, loss_fonction=BCELoss(),nb_iteration=n_iter)

plt.figure()
plt.plot(range(1,n_iter+1),loss)
plt.xlabel("nombre d'itération")
plt.ylabel('cout')
plt.show()

idx = np.random.choice(len(alltestx), 5)

for i in idx:
    plt.figure()
    plt.title('image originale de '+str(alltesty[i]))
    plt.imshow(alltestx[i].reshape(16,16))
    plt.show()

    plt.figure()
    plt.title('image bruitée de '+str(alltesty[i]))
    plt.imshow(alltestx_noise[i].reshape(16, 16))
    plt.show()

    Xhat = seq.forward(alltestx)[-1]

    plt.figure()
    plt.title('image reconstruite de ' + str(alltesty[i]))
    plt.imshow(Xhat[i].reshape(16,16))
    plt.show()