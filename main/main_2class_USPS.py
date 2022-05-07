import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from data.USPS_tools import load_usps, get_usps, show_usps
from encapsulage import Sequentiel, SGD
from loss.ce import SMCELoss
from modules.convolution import ReLu
from modules.linear import Linear
from modules.nonlinear import TanH

if __name__ =="__main__":
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    datax,datay=get_usps([0,1],alltrainx,alltrainy)
    testx, testy = get_usps([0,1], alltestx, alltesty)
    datay = OneHotEncoder(sparse=False).fit_transform(datay.reshape(-1, 1))
    print(len(datax))
    testy = OneHotEncoder(sparse=False).fit_transform(testy.reshape(-1, 1))
    print(len(testx))
    lin1 = Linear(256, 256, biais=True)
    lin2 = Linear(256, 10)
    seq = Sequentiel([lin1,ReLu(),lin2,TanH()])
    n_iter = 100
    eps = 1e-3
    loss = SGD(seq, datax, datay, 100, n_iter, SMCELoss(), eps)

    plt.figure()
    plt.plot(loss)
    plt.xlabel('n_iter')
    plt.ylabel('cout')
    plt.show()

    outputs = seq.forward(testx)
    print(len(outputs))
    yhat = np.argmax(outputs[-1], axis=1)

    idx = np.random.choice(len(testx), 4)
    for i in idx:
        plt.figure()
        plt.title('y: ' + str(testy[i]) + ' yhat :' + str(yhat[i]))
        show_usps(testx[i])