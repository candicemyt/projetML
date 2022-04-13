from linear import Linear
from mse import MSELoss
from nonlinear import *
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt

#test de l'implémentation avec descente de gradient
datax = np.random.random(size=(10,4))
datay = np.random.randint(0,2, (10,))

#todo faire fonctionner le test avec les couches de l'enonce partie 2

mse = MSELoss()
lin1 = Linear(datax.shape[1], 3)
lin2 = Linear(3, 1)
tanh = TanH()
sigm = Sigmoide()

val_mse = []
for i in range(100):
     output1 = lin1.forward(datax)
     output2 = tanh.forward(output1)
     output3 = lin2.forward(output2)
     output4 = sigm.forward(output3)
     loss = mse.forward(output4, datay).mean()
     val_mse.append(loss)
     delta = mse.backward(output4, datay)
     delta = sigm.backward_delta(output3, delta)
     lin2.zero_grad()
     lin2.backward_update_gradient(output2, delta)
     lin2.update_parameters(gradient_step=1e-2)
     delta = lin2.backward_delta(output2, delta)
     delta = tanh.backward_delta(output1, delta)
     lin1.zero_grad()
     lin1.backward_update_gradient(datax, delta)
     lin1.update_parameters(gradient_step=1e-2)





plt.plot(val_mse)
plt.show()

