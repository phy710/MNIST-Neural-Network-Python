'''
A simple neural netwrok for MNIST
Author: Zephyr
10/20/2018
'''
import numpy as np
from matplotlib import pyplot
import time;
import loadMNIST
import nn

if __name__ == '__main__':
    time_start = time.time()

    etaDecay = 0.9
    epochMax = 10
    opt = nn.optmizer(lr=0.5, decay=0.9, batchSize=100, loss='MSE', l2=0.001)
    
    model = []
    model.append(nn.layer(784, 100, 'tanh'))
    model.append(nn.layer(model[0].neuronsNum, 100, 'tanh'))
    model.append(nn.layer(model[1].neuronsNum, 10, 'sigmoid'))                   

    trainData = loadMNIST.loadMNISTImages('train-images.idx3-ubyte')
    trainLabels = loadMNIST.loadMNISTLabels('train-labels.idx1-ubyte')
    testData = loadMNIST.loadMNISTImages('t10k-images.idx3-ubyte')
    testLabels = loadMNIST.loadMNISTLabels('t10k-labels.idx1-ubyte')
    trainProbabilities = np.zeros((10, trainLabels.size));
    for a in range(trainLabels.size):
        trainProbabilities[trainLabels[a], a] = 1
    testProbabilities = np.zeros((10, testLabels.size));
    for a in range(testLabels.size):
        testProbabilities[testLabels[a], a] = 1

    estimatedTrainProbabilities = nn.ff(model, trainData)
    trainLoss = np.zeros(epochMax+1)
    trainAcc = np.zeros(epochMax+1)
    trainLoss[0], trainAcc[0] = nn.evaluate(model, opt, trainData, trainProbabilities)    
    testLoss = np.zeros(epochMax+1)
    testAcc = np.zeros(epochMax+1)
    testLoss[0], testAcc[0] = nn.evaluate(model, opt, testData, testProbabilities)    
    for epoch in range(1, epochMax+1):
        nn.train(model, opt, trainData, trainProbabilities)
        trainLoss[epoch], trainAcc[epoch] = nn.evaluate(model, opt, trainData, trainProbabilities)    
        testLoss[epoch], testAcc[epoch] = nn.evaluate(model, opt, testData, testProbabilities)    
        print('Epoch:', epoch, ', Learning Rate:', opt.lr, ', Loss:', trainLoss[epoch], '/', testLoss[epoch], 'Acc:', trainAcc[epoch], '/', testAcc[epoch])
        if trainLoss[epoch]>trainLoss[epoch-1]:
            opt.lrDecay();

    time_end = time.time()
    print('Elapsed time:', time_end-time_start, 'seconds.')
    print('Training accuracy:', trainAcc[-1]*100, '%;')
    print('Test accuracy:', testAcc[-1]*100, '%.')
    
    fig1 = pyplot.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(range(epochMax+1), 100*trainAcc, label='Training')
    ax1.plot(range(epochMax+1), 100*testAcc, label='Test')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, epochMax)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy')

    fig2 = pyplot.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(range(epochMax+1), trainLoss, label='training')
    ax2.plot(range(epochMax+1), testLoss, label='test')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, epochMax)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(opt.loss)
    ax2.set_title('Loss')
    
    pyplot.show(block=False)
