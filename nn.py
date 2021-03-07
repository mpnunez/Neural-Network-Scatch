#! ./.venv/bin/python

from mnist import load_mnist

import numpy as np

def relu(x):
    return 0 if x < 0 else x

vrelu = np.vectorize(relu)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

vsigmoid = np.vectorize(sigmoid)

def softmax(x):
    exps = [np.exp(i) for i in x]
    total = np.sum(exps)
    return np.array([e / total for e in exps])

class Layer:
    def __init__(self,n_input,n_output):
        self.weights = np.zeros([n_output,n_input])
        self.biases = np.zeros(n_output)
        self.activation = vsigmoid

    def randomize(self):
        self.weights = np.random.rand(*self.weights.shape)
        self.biases = np.random.rand(self.weights.shape[0])

    def process(self,z):
        print(z.shape)
        print(self.weights.shape)
        print(self.biases.shape)
        return self.activation(np.matmul(self.weights, z) + self.biases)

class FeedForwardNeuralNetwork:
    def __init__(self):
        pass

def main():
    print("Hello World")
    (train_X, train_Y), (test_X, test_Y) = load_mnist()

    l = Layer(28*28,10)
    l.randomize()
    l.activation = softmax
    y = l.process(train_X[0].flatten())

    print(y)



if __name__ == "__main__":
    main()
