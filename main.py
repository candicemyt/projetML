from linear import Linear, MSELoss
import numpy as np


#test de l'implémentation en réalisant une boucle d'apprentissage par descente de gradient
datax = np.random.randn(10,10)
print(datax)
datay = np.random.choice([-1,1],10)
print(datay)
mse = MSELoss()
linear = Linear(10,1)
res_lin = linear.forward(datax)
res_mse = mse.forward(datay.reshape(-1,1), res_lin)
delta_mse = mse.backward(datay.reshape(-1,1),res_lin)
linear.backward_update_gradient(datax,delta_mse)
grad_lin = linear._gradient
delta_lin = linear.backward_delta(datax,delta_mse)