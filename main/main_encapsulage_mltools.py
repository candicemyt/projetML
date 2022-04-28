from encapsulage import *
from modules.linear import *
from loss.mse import *
from modules.nonlinear import *
from mltools import *

datax, datay = gen_arti(epsilon=0.1, data_type=0)
datay = datay.reshape(-1, 1)

lin1 = Linear(2, 10,biais=True)
lin2 = Linear(10, 1)
seq=Sequentiel([lin1, TanH(), lin2, Sigmoide()])
eps = 1e-4
n_iter = 1000

loss=SGD(seq,datax,datay,100,n_iter,MSELoss(),eps)

def predict(datax):
    seq.forward(datax)
    outputs=seq.outputs
    return np.where(outputs[-1] > 0.5, 1, -1)

plt.figure()
plt.plot(loss)
plt.xlabel('n_iter')
plt.ylabel('cout')
plt.show()

plt.figure()
plt.title('No linear')
plot_frontiere(datax, lambda x: predict(x), step=100)
plot_data(datax, datay.reshape(-1))
plt.show()