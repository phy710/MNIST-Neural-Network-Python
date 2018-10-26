import numpy as np

def act(x, method='none'):
    # Activate function tanh
    if method == 'tanh':
        return np.tanh(x)
    elif method == 'sigmoid':
        return 1/(1+np.exp(-x))
    elif method == 'softmax':
        return np.exp(x)/sum(np.exp(x))
    elif method == 'ReLU' or method == 'relu' or method == 'reLU':
        return (abs(x)+x)/2
    elif method == 'none':
        return x

def dact(x, method='none'):
    # Derivate of activate function tanh
    if method == 'tanh':
        return np.ones(x.shape) - np.tanh(x)**2
    elif method == 'sigmoid':
        return np.exp(-x)/(1+np.exp(-x))**2
    elif method =='ReLU' or method == 'relu' or method == 'reLU':
        return 1*(x>0)+0.5*(x==0)
    elif method == 'none':
        return np.ones(x.shape)

def loss(d, y, opt):
    if opt.loss == 'MSE' or opt.loss == 'mse':
        return np.mean((d-y)**2)
    elif opt.loss == 'cross-entropy' or opt.loss == 'cross entropy' or opt.loss == 'crossEntropy':
        return -np.mean(d*np.log(y))
            

def ff(model, x):
    model[0].output_ = np.dot(model[0].weights, x) + model[0].biases
    model[0].output = act(model[0].output_, model[0].act)
    if len(model)>=2:
        for a in range(1, len(model)):
            model[a].output_ = np.dot(model[a].weights, model[a-1].output) + model[a].biases
            model[a].output = act(model[a].output_, model[a].act)
    return model[-1].output

def bp(model, opt, x, d):
    if opt.loss == 'MSE' or opt.loss == 'mse':
        delta = (d-model[-1].output)*dact(model[-1].output_, model[-1].act)
    elif opt.loss == 'cross-entropy' or opt.loss == 'cross entropy' or opt.loss == 'crossEntropy':
        assert model[-1].act=='softmax', 'Activation function of final layer should be softmax for cross-entropy loss function!'
        delta = d-model[-1].output
    for a in range(len(model)-1, 0, -1):
        model[a].weights += opt.lr*np.dot(delta, model[a-1].output.T)/opt.batchSize - opt.lr*opt.l2*model[a].weights/opt.batchSize
        model[a].biases += np.mean(opt.lr*delta) - -opt.lr*opt.l2*model[a].biases
        delta = np.dot(model[a].weights.T, delta)*dact(model[a-1].output_, model[a-1].act)
    model[0].weights += opt.lr*np.dot(delta, x.T)/opt.batchSize - opt.lr*opt.l2*model[0].weights
    model[0].biases += np.mean(opt.lr*delta) -opt.lr*opt.l2*model[0].biases
        
def train(model, opt, x, d):
    n = x.shape[1]
    assert np.mod(n, opt.batchSize)==0, "Batch number is not an integer!"
    batchNum = n//opt.batchSize
    k = np.random.permutation(x.shape[1])
    for a in range(batchNum):
        ff(model, x[:, k[a*opt.batchSize:(a+1)*opt.batchSize]])
        bp(model, opt, x[:, k[a*opt.batchSize:(a+1)*opt.batchSize]], d[:, k[a*opt.batchSize:(a+1)*opt.batchSize]])
        

def evaluate(model, opt, x, d):
    y = ff(model, x)
    l = loss(d, y, opt)
    acc = np.mean(y.argmax(0) == d.argmax(0))
    return l, acc

class optmizer:
    def __init__(self, lr=0.01, decay=1, batchSize=1, loss='MSE', l2=0):
        self.lr = lr;
        self.decay = decay
        self.batchSize = batchSize
        self.l2 = l2;
        self.loss = loss
    def lrDecay(self):
        self.lr *= self.decay

class layer:
    def __init__(self, inputNum, neuronsNum, act):
        self.weights = np.random.uniform(-0.1, 0.1, (neuronsNum, inputNum))
        self.biases = np.random.uniform(-0.1, 0.1, (neuronsNum, 1))
        self.neuronsNum = neuronsNum
        self.act = act
        self.output_ = np.zeros((neuronsNum, 1))
        self.output = np.zeros((neuronsNum, 1))
