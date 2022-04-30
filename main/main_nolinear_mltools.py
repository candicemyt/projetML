from modules.linear import Linear
from loss.mse import MSELoss
from modules.nonlinear import *
import numpy as np
from data.mltools import plot_data, plot_frontiere, gen_arti
import matplotlib.pyplot as plt


datax, datay = gen_arti(epsilon=0.1, data_type=0)
datay = datay.reshape(-1, 1)

lin1 = Linear(2, 10,biais=True)
lin2 = Linear(10, 1)
modu = [lin1, TanH(), lin2, Sigmoide()]
n_iter = 1000
eps = 1e-4
loss = []
for it in range(n_iter):
    outputs = [datax]
    for mod in modu:
        outputs.append(mod.forward(outputs[-1]))
    tmp_loss = MSELoss().forward(datay, outputs[-1])
    tmpDelta = MSELoss().backward(datay, outputs[-1])
    for i in range(len(outputs) - 2, -1, -1):
        module = modu[i]
        delta = module.backward_delta(outputs[i], tmpDelta)
        module.backward_update_gradient(outputs[i], tmpDelta)
        module.update_parameters(gradient_step=eps)
        module.zero_grad()
        tmpDelta = delta

    loss.append(tmp_loss.mean())


def predict(datax):
    outputs = [datax]
    for mod in modu:
        outputs.append(mod.forward(outputs[-1]))
    return np.where(outputs[-1] > 0.5, 1, -1)


plt.figure()
plt.plot(range(1, n_iter + 1), loss)
plt.xlabel("nombre d'itération")
plt.ylabel('cout')
plt.show()

plt.figure()
plt.title('No linear')
plot_frontiere(datax, lambda x: predict(x), step=100)
plot_data(datax, datay.reshape(-1))
plt.show()
