"""
Activation functions
"""

import numpy as np

def relu(x):
    return 0 if x < 0 else x

vrelu = np.vectorize(relu)

def relu_grad(x):
    return 0 if x < 0 else 1

vrelu_grad = np.vectorize(relu_grad)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

vsigmoid = np.vectorize(sigmoid)

def sigmoid_grad(x):
    return np.exp(x) / (1 + np.exp(-x)) ** 2

vsigmoid_grad = np.vectorize(sigmoid_grad)

def softmax(x):
    exps = [np.exp(i) for i in x]
    total = np.sum(exps)
    return np.array([e / total for e in exps])

def softmax_grad(x):
    exps = [np.exp(i) for i in x]
    total = np.sum(exps)
    return np.array([(total * e - e ** 2) / total**2 for e in exps]) / total**2
