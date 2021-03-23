from activation import softmax, sigmoid
import numpy as np

def test_sigmoid():
    x = np.array([-4,-3,-2,-1,-0,1,2,3,4])
    value, grad = sigmoid(x)

def test_softmax():
    x = np.array([1,2,3,4,5,-40])
    value, grad = softmax(x)

    print(value)
    print(grad)

    assert np.abs(np.sum(value) - 1) < 0.001
    assert np.abs(np.sum(grad)) < 0.001
