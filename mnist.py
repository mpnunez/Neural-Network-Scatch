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
