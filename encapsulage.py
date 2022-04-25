from mse import MSELoss

class Sequentiel(object):

    def __init__(self, modules):
        self.modules = modules
        self.outputs = []

    def forward(self, datax):

        self.outputs.append(self.modules[0].forward(datax))
        for m in self.modules[1::]:
            input = self.outputs[-1]
            self.outputs.append(m.forward(input))

    def backward(self, datax, delta):

        inputs = self.outputs[::-1]
        modules_bckw = self.modules[::-1]

        for i in range(len(inputs) - 1):
            modules_bckw[i].backward_update_gradient(inputs[i + 1], delta)
            delta = modules_bckw[i].backward_delta(inputs[i + 1], delta)

        modules_bckw[-1].backward_update_gradient(datax, delta)
        modules_bckw[-1].backward_delta(datax, delta)
        self.outputs = []
        return inputs[0]



class Optim(object):

    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.loss_values = []

    def step(self, datax, datay):
        self.net.forward(datax)
        self.loss_values.append(self.loss.forward(self.net.outputs[-1], datay).mean())
        delta = self.loss.backward(datay, self.net.outputs[-1])
        self.net.backward(datax, delta)
        for m in self.net.modules:
            m.update_parameters(gradient_step=self.eps)

# def sgd(net,datax,datay,batch_size,nb_iter):
#     mse = MSELoss()
#     opti=Optim(net,mse,1e-4)
#     datax=datax.reshape(batch,len(datax)/batch)