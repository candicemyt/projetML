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

def SGD(net, datax, datay, batch_size, nb_iter, loss_fonction, eps):
    op = Optim(net, loss_fonction, eps)
    sum_loss=[]
    indexes = np.arange(len(datax))
    #on crée les batch de manière aléatoire
    N_batch = len(datax) // batch_size
    batchs=[]
    for b in range(N_batch):
        a=np.random.choice(indexes,batch_size,replace=False)
        indexes=np.setdiff1d(indexes,a)
        dataxb = np.array([datax[i] for i in a])
        datayb = np.array([datay[i] for i in a])
        batchs.append((dataxb,datayb))

    for i in range(nb_iter):

        for (batchx,batchy) in batchs:
            op.step(batchx, batchy)
        sum_loss.append(np.mean(op.loss_values[i*N_batch:(i+1)*N_batch]))
        if i%(nb_iter//10)==0:
            print("interation ",i," : ",np.mean(op.loss_values[i*N_batch:(i+1)*N_batch]))
    return sum_loss