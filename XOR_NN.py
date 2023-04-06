import numpy as np
def fit(x_train, y_train, learningRate):
    eta = 1
    n_samples, dim = x_train.shape
    weight = np.zeros(dim)
    bias = 0

    for i in range(n_samples):
        Y = Y_in(weight, x_train[i, :]) + bias
        threshold = step(Y)
        error = target - threshold
        for j in range(dim):
            weight[j] = eta * error * x_train[i][j]
            bias = bias + lr * error

def Y_in(x, w):
    z = np.dot(x, w)
    return z

def step(z):
    if z == 0:
        return 0
    elif z < -1:
        return -1
    else:
        return 1

x = np.array([[1, 1], [-1, 1], [1, -1], [-1, 1]])
target = np.array([[1, -1, -1, -1]])
lr = 1
weight, bias = fit(x, target, lr)