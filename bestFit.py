import matplotlib.pyplot as plt
import numpy as np
import gradientAscent as ga

def plotBestFit():

    #weights = wei.getA()
    dataMat,labelMat=ga.loadDataSet()
    #weights = ga.gradAscent(dataMat,labelMat)
    #weights = ga.stocGradAscent0(dataMat,labelMat)
    weights = ga.stocGradAscent1(dataMat, labelMat)
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = np.array((-weights[0]-weights[1]*x)/weights[2]).transpose()
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2');
    plt.show()

plotBestFit()