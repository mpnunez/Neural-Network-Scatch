#! ./.venv/bin/python

from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt

def load_mnist():
    
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

    train_X = np.array(train_X) / 256
    test_X = np.array(test_X) / 256

    def convert_y(y_data):
        y_lol = [[int(i == j) for j in range(10)] for i in y_data]
        return np.array(y_lol)

    train_Y = convert_y(train_Y)
    test_Y = convert_y(test_Y)

    print(test_Y[-2::])


    plt.imshow(train_X[0])
    plt.show()

def main():
    print("Hello World")
    load_mnist()



if __name__ == "__main__":
    main()
