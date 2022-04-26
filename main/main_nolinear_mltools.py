from linear import Linear
from mse import MSELoss
from nonlinear import *
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt

def proj_biais(datax):
    return np.hstack((np.ones(datax.shape[0]).reshape(-1,1),datax))

datax, datay = gen_arti(epsilon=0.1,data_type=1)
datay = datay.reshape(-1,1)

datax_biais = proj_biais(datax)

lin1 = Linear(3, 10)
lin2 = Linear(10, 1)
modu = [lin1, TanH(), lin2, Sigmoide()]
n_iter = 1000
eps = 1e-4
loss = []
for it in range(n_iter):
      outputs = [datax_biais]
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
      if it % 20 == 0:
            print("iteration", it, "loss =", np.mean(loss))


def predict(datax):
      datax_b = proj_biais(datax)
      outputs = [datax_b]
      for mod in modu:
            outputs.append(mod.forward(outputs[-1]))
      return np.where(outputs[-1] > 0.5, 1, -1)

plt.figure()
plt.plot(range(1,n_iter+1),loss)
plt.xlabel("nombre d'itération")
plt.ylabel('cout')
plt.show()

plt.figure()
plt.title('No linear')
plot_frontiere(datax,lambda x : predict(x),step=100)
plot_data(datax,datay.reshape(-1))
plt.show()