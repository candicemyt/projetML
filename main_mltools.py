from mltools import *
from linear import Linear
from mse import MSELoss
import matplotlib.pyplot as plt

#TO DO : loss croissante 

datax, datay = gen_arti(data_type=0,epsilon=0.1,nbex=1000)

mse = MSELoss()
lin = Linear(2, 1)
val_mse = []
all_w = []

for i in range(100):
    output = lin.forward(datax)
    loss = mse.forward(output, datay).mean()
    val_mse.append(loss)
    delta_mse = mse.backward(datay, output)
    lin.zero_grad()
    lin.backward_update_gradient(datax, delta_mse)
    lin.update_parameters(gradient_step=1e-2)
    all_w.append(lin._parameters.copy())
w_star=lin._parameters.copy()

## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
grid, x_grid, y_grid = make_grid(xmin=-2, xmax=2, ymin=-2, ymax=2, step=100)

## Visualisation des données et de la frontière de décision
plt.figure()
plot_frontiere(datax,lambda x : np.sign(x.dot(w_star)),step=100)
plot_data(datax, datay)
plt.show()

## Visualisation de la fonction de coût en 2D
plt.figure()
plt.title("fonction de coût")
plt.plot(val_mse)
plt.show()

print(all_w)

