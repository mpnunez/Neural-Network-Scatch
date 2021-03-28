import numpy as np

def categorical_cross_entropy_loss(y_pred, y_actual):
    """
    'Cross-entropy' for categorical values with only 0 or 1 in y

    y_actual : vector of actual probabilities (may be [0,...,1,...,0])
    y_pred : vector of probabilities

    Need to also return derivative of loss for each y

    """

    loss = -1 * np.sum( y_actual * np.log(y_pred), axis=int(len(y_pred.shape)==2) )
    J = - y_actual / y_pred

    return loss, J
