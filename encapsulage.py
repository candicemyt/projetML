import numpy as np


class Sequentiel(object):

    def __init__(self, modules):
        self.modules = modules
        self.outputs = []

    def forward(self, datax):
        self.outputs=[datax]
        for m in self.modules:
            self.outputs.append(m.forward(self.outputs[-1]))

    def backward(self, tmpDelta,eps):
        for i in range(len(self.outputs) - 2, -1, -1):
            module = self.modules[i]
            delta = module.backward_delta(self.outputs[i], tmpDelta)
            module.backward_update_gradient(self.outputs[i], tmpDelta)
            module.update_parameters(gradient_step=eps)
            module.zero_grad()
            tmpDelta = delta

class Optim(object):

    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps
        self.loss_values = []

    def step(self, datax, datay):
        self.net.forward(datax)
        outputs = self.net.outputs
        tmp_loss = self.loss.forward(datay, outputs[-1])
        tmpDelta = self.loss.backward(datay, outputs[-1])
        self.net.backward(tmpDelta, self.eps)
        self.loss_values.append(tmp_loss.mean())

def SGD(net,datax,datay,batch_size,nb_iter,loss_fonction,eps):
    op = Optim(net, loss_fonction, eps)
    for epch in range(nb_iter):
        indexes = np.random.randint(0, len(datax), batch_size)  # random sample
        dataxb = np.array([datax[i] for i in indexes])
        datayb = np.array([datay[i] for i in indexes])
        op.step(dataxb, datayb)
    return op.loss_values