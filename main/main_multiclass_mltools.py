import numpy as np

from data.USPS_tools import load_usps, get_usps

if __name__ =="__main__":
    uspsdatatrain = "../data/USPS_train.txt"
    uspsdatatest = "../data/USPS_test.txt"
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    neg = 5
    pos = 6
    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    datay = np.where(datay==5,1,1)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)