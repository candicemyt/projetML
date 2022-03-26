from linear import Linear, MSELoss
import numpy as np
from mltools import plot_data, plot_frontiere, make_grid, gen_arti
import matplotlib.pyplot as plt

#test de l'implémentation en réalisant une boucle d'apprentissage par descente de gradient
"""datax = np.random.randn(10,10)
print(datax)
datay = np.random.choice([-1,1],10)
print(datay)"""
datax, datay = gen_arti(data_type=0,epsilon=0.1,nbex=2000)
print(datax.shape)
print(datay.shape)
plt.figure()
plot_data(datax,datay)
plt.show()

mse = MSELoss()
linear = Linear(2000,2)

#res_lin = linear.forward(datax)
#res_mse = mse.forward(datay.reshape(-1,1), res_lin)
#delta_mse = mse.backward(datay.reshape(-1,1),res_lin)
#linear.backward_update_gradient(datax,delta_mse)
#grad_lin = linear._gradient
#delta_lin = linear.backward_delta(datax,delta_mse)

allw = [linear._parameters]
val_mse = []
for i in range(100):
    res_lin = linear.forward(datax)
    val_mse.append(mse.forward(datay.reshape(-1, 1), res_lin))
    delta_mse = mse.backward(datay.reshape(-1, 1), res_lin)
    linear.backward_update_gradient(datax, delta_mse)
    grad_lin = linear._gradient
    delta_lin = linear.backward_delta(datax, delta_mse)
    linear.update_parameters()
    allw.append(linear._parameters)
w=linear._parameters

