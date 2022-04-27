from encapsulage import *
from modules.linear import *
from loss.mse import *
from modules.nonlinear import *
from mltools import *

def proj_biais(datax):
    return np.hstack((np.ones(datax.shape[0]).reshape(-1, 1), datax))

datax, datay = gen_arti(epsilon=0.1, data_type=1)
datay = datay.reshape(-1, 1)

datax_biais = proj_biais(datax)

lin1 = Linear(3, 10)
lin2 = Linear(10, 1)
seq=Sequentiel([lin1, TanH(), lin2, Sigmoide()])
eps = 1e-4
n_iter = 1000

loss=SGD(seq,datax_biais,datay,100,n_iter,MSELoss(),eps)

def predict(datax):
    datax_biais = proj_biais(datax)
    seq.forward(datax_biais)
    outputs=seq.outputs
    return np.where(outputs[-1] > 0.5, 1, -1)

plt.figure()
plt.plot(loss)
plt.xlabel("batch")
plt.ylabel('cout')
plt.show()

plt.figure()
plt.title('No linear')
plot_frontiere(datax, lambda x: predict(x), step=100)
plot_data(datax, datay.reshape(-1))
plt.show()