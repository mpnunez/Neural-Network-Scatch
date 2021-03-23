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
        self.layers = []
        self.learning_rate = 0.01

    def compute_data_point(self,x):

        z_outputs = [None for l in self.layers]
        z_grads = [None for l in self.layers]

        for l in self.layers:
            x, gradx = l.process(x)
        return x

    def process(self,x_mat,y_mat):

        return np.array([compute_data_point(self.x_vec) for x_vec in x_mat])


def loss(y_pred, y_actual):
    """
    y_actual : vector of actual probabilities (may be [0,...,1,...,0])
    y_pred : vector of probabilityies
    """
    return -1 * np.sum( y_actual * np.log(y_pred), axis=1 )

def main():
    print("Hello World")
    (train_X, train_Y), (test_X, test_Y) = load_mnist()

    """
    Build neural network architecture
    """

    l1 = Layer(28*28,32)
    l1.randomize()

    l2 = Layer(32,10)
    l2.randomize()
    l2.activation = softmax
    l2.activation_grad = softmax_grad

    nn = FeedForwardNeuralNetwork()
    nn.layers = [l1,l2]

    """
    Evaluate a few data points
    """

    x_data = train_X[0:5]
    y_data = train_Y[0:5]

    p = nn.process(x_data)
    my_loss = loss(p, y_data)

    print(p)
    print(my_loss)
    #print(sum(p))



if __name__ == "__main__":
    main()
