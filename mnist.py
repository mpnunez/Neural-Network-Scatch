from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt

def load_mnist():

    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

    train_X = np.array(train_X) / 256
    test_X = np.array(test_X) / 256

    def convert_x(x_data):
        return np.array( [i.flatten() for i in x_data] )

    def convert_y(y_data):
        return np.array([[int(i == j) for j in range(10)] for i in y_data])

    train_X = convert_x(train_X)
    test_X = convert_x(test_X)
    train_Y = convert_y(train_Y)
    test_Y = convert_y(test_Y)
    return (train_X, train_Y), (test_X, test_Y)

def train_mnist():
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

    nn = FeedForwardNeuralNetwork(
        batch_size = 10,
        epochs = 1000,
        learning_rate = 0.1)
    nn.layers = [l1,l2]

    """
    Evaluate a few data points
    """

    nn.train(train_X, train_Y)
    # test_pred_Y = nn.predict(test_X)
    # test_pred_Y = np.round(test_pred_Y)
    # frac_wrong = np.sum(test_pred_Y - test_Y)
    # print("Fraction wrong: {}".format(frac_wrong))
