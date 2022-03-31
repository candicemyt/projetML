from linear import Linear, MSELoss
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt

#test de l'implémentation en réalisant une boucle d'apprentissage par descente de gradient
datax = np.array([1.,2.,1.,0.])
datay = np.array([1.,0.,1.])

mse = MSELoss()
linear = Linear(datax.shape[0],2)

allw = [linear._parameters]
val_mse = []
for i in range(10):
    res_lin = linear.forward(datax)
    print(res_lin.shape)
    print(datay.shape)
    val_mse.append(mse.forward(datay, res_lin).mean())
    delta_mse = mse.backward(datay, res_lin)
    linear.zero_grad()
    linear.backward_update_gradient(datax, delta_mse)
    grad_lin = linear._gradient
    delta_lin = linear.backward_delta(datax, delta_mse)
    linear.update_parameters()
    allw.append(linear._parameters.copy())
w=linear._parameters


plt.plot(val_mse)
plt.show()

