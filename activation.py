"""
Activation functions
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function
    """

    exps = np.exp(-x)
    final = 1 / (1 + exps)
    J = np.diag(exps / (1 + exps) ** 2)

    return final, J


def softmax(x):
    """
    Softmax activation function
    """

    exps = np.exp(x)
    exps_sum = np.sum(exps)
    final = exps / exps_sum
    J = (exps_sum * np.diag(exps) - np.outer(exps,exps) ) / exps_sum ** 2

    return final, J
