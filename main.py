from linear import Linear
from mse import MSELoss
from nonlinear import *
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt
from encapsulage import Sequentiel, Optim

#todo faire marcher optim (loss très bizzare)

#test de l'implémentation avec descente de gradient
datax = np.random.random(size=(10,4))
datay = np.random.randint(0,2, (10,3))


mse = MSELoss()
lin1 = Linear(datax.shape[1], 5)
lin2 = Linear(5, datay.shape[1])
tanh = TanH()
sigm = Sigmoide()
seq = Sequentiel([lin1, tanh, lin2, sigm])
optim = Optim(seq, mse, 1e-7)
val_mse = []
for i in range(1000):
     optim.step(datax, datay)
     val_mse.append(optim.loss_values)

     # seq.forward(datax)
     # delta = mse.backward(datay, seq.outputs[-1])
     # seq.backward(datax, delta)
     #


#      output1 = lin1.forward(datax)
#      output2 = tanh.forward(output1)
#      output3 = lin2.forward(output2)
#      output4 = sigm.forward(output3)
#      loss = mse.forward(output4, datay).mean()
#      val_mse.append(loss)
#
#      delta = mse.backward(datay, output4)
#      delta = sigm.backward_delta(output3, delta)
#      lin2.zero_grad()
#      lin2.backward_update_gradient(output2, delta)
#      delta = lin2.backward_delta(output2, delta)
#      lin2.update_parameters(gradient_step=1e-4)
#      delta = tanh.backward_delta(output1, delta)
#      lin1.zero_grad()
#      lin1.backward_update_gradient(datax, delta)
#      delta = lin1.backward_delta(datax, delta)
#      lin1.update_parameters(gradient_step=1e-4)
# print(output4)
# print(datay)

print(val_mse[0])
plt.plot(val_mse)
plt.show()

