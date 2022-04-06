from linear import Linear, MSELoss
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt

#test de l'implémentation avec descente de gradient
datax = np.random.random(size=(10,4))
datay = np.random.randint(0,2, (10,))

mse = MSELoss()
linear = Linear(datax.shape[1],1)

val_mse = []
for i in range(100):
     output = linear.forward(datax)
     loss = mse.forward(output, datay).mean()
     val_mse.append(loss)
     delta_mse = mse.backward(output, datay)
     linear.zero_grad()
     linear.backward_update_gradient(datax, delta_mse)
     linear.update_parameters(gradient_step=1e-2)

plt.plot(val_mse)
plt.show()

