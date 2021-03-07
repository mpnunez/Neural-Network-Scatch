#! ./.venv/bin/python


import numpy as np

from mnist import load_mnist
from activation import softmax_grad, softmax, vsigmoid, vsigmoid_grad

class Layer:
    def __init__(self,n_input,n_output):
        self.weights = np.zeros([n_output,n_input])
        self.biases = np.zeros(n_output)
        self.activation = vsigmoid
        self.activation_grad = vsigmoid_grad

    def randomize(self):
        self.weights = np.random.rand(*self.weights.shape)
        self.biases = np.random.rand(self.weights.shape[0])

    def process(self,z):
        z_new = np.matmul(self.weights, z) + self.biases
        return self.activation(z_new), self.activation_grad(z_new)

class FeedForwardNeuralNetwork:
    def __init__(self):
        pass

def main():
    print("Hello World")
    (train_X, train_Y), (test_X, test_Y) = load_mnist()

    l = Layer(28*28,10)
    l.randomize()
    l.activation = softmax
    l.activation_grad = softmax_grad
    y, y_grad = l.process(train_X[0].flatten())

    print(y)
    print(y_grad)
    print(np.sum(y))
    print(np.sum(y_grad))



if __name__ == "__main__":
    main()
