#! ./.venv/bin/python

from mnist import load_mnist

import numpy as np

def relu(x):
    return 0 if x < 0 else x

vrelu = np.vectorize(relu)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

vsigmoid = np.vectorize(sigmoid)


class Layer:
    def __init__(self,n_input,n_output):
        self.weights = np.zeros([n_input,n_output])
        self.biases = np.zeros(n_output)
        self.activation = "RELU"

    def randomize():
        self.weights = np.random.rand(n_input,n_output)
        self.biases = nprandom.rand(n_output)

    def process(z):
        np.matmul(self.weights, z) + self.biases

class FeedForwardNeuralNetwork:
    def __init__(self):
        pass

def main():
    print("Hello World")
    (train_X, train_Y), (test_X, test_Y) = load_mnist()

    print(vsigmoid(train_X[0].flatten()))

    #l = Layer(28*28,10)
    #l.randomize()
    #y = l.process(train_X[0].flatten())

    #print(vsigmoidy)



if __name__ == "__main__":
    main()
