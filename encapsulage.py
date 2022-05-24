import numpy as np

class Sequentiel:
    def __init__(self,modules):
        self.modules = modules

    def forward(self, input):
        inputs = [input]
        for module in self.modules:
            inputs.append(module.forward(input))
            input = inputs[-1]
        return inputs

    def backward(self, outputs, lastDelta, eps):
        tmpDelta = lastDelta
        for i in range(len(outputs)-2,-1, -1):
            module = self.modules[i]
            delta = module.backward_delta(outputs[i], tmpDelta)
            module.backward_update_gradient(outputs[i], tmpDelta)
            module.update_parameters(gradient_step = eps)
            module.zero_grad()
            tmpDelta = delta

class Optim:
    def __init__(self,net,loss,eps):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self,datax,datay):
        outputs = self.net.forward(datax)
        loss = self.loss.forward(datay,outputs[-1])
        lastDelta = self.loss.backward(datay,outputs[-1])
        self.net.backward(outputs, lastDelta, self.eps)
        return loss.mean()

def mini_SGD(net, datax, datay ,batch_size, nb_iteration , loss_fonction , eps = 1e-5):
    opt = Optim(net, loss_fonction, eps)
    sum_loss = []
    N_batch = len(datax) // batch_size
    for it in range(nb_iteration):
        loss = []
        for epch in range(N_batch):
            a = np.random.choice(datax.shape[0], batch_size, replace=False)
            x, y = datax[a], datay[a]
            tmp_loss = opt.step(x,y)
            loss.append(tmp_loss)
        sum_loss.append(np.mean(loss))
        #if it % 20 == 0:
            #print("iteration", it, "loss =", np.mean(loss))

         
    return net, sum_loss
