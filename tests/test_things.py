import numpy as np

from activation import softmax, sigmoid
from loss import categorical_cross_entropy_loss


def test_sigmoid():
    x = np.array([-4,-3,-2,-1,-0,1,2,3,4])
    value, grad = sigmoid(x)

    print("\nSigmoid results")
    print(value)
    print(grad)


def test_softmax():
    x = np.array([1,2,3,4,5,-40])
    value, grad = softmax(x)

    print("\nSoftmax results")
    print(value)
    print(grad)

    assert np.abs(np.sum(value) - 1) < 0.001
    assert np.abs(np.sum(grad)) < 0.001

def test_loss():

    # Random numbers adding to 1 in each row
    y_pred = np.random.rand(10,5)
    y_pred = [row / np.sum(row) for row in y_pred]

    # One 1 in each row
    y_actual = np.array([[int(i == int(j/2)) for i in range(5)] for j in range(10)])

    print("\nLoss results")
    print(y_pred)
    print(y_actual)

    l, j = categorical_cross_entropy_loss(y_pred, y_actual)
    print(l)
    print(j)
