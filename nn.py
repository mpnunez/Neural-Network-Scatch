#! ./.venv/bin/python


import numpy as np

from mnist import load_mnist
from loss import categorical_cross_entropy_loss
from activation import softmax, sigmoid
from tqdm import tqdm

class Layer:
    """
    Neural network layer
    """

    def __init__(self,n_input,n_output):

        # Add 1 to input nodes for biases
        self.weights = np.zeros([n_output,n_input+1])
        self.loss_grads = np.zeros(self.weights.shape)
        self.activation = sigmoid


    def randomize(self):
        """
        Initialize weights with random numbers between 0 and 1
        """
        self.weights = np.random.rand(*self.weights.shape)

    def propagate(self,z):
        """
        Given output values from the previous layer,
        compute the output and derivative from this layer

        z : vector
            Output values from previous layer
        """

        z_new = np.matmul(self.weights, np.append(z, 1))
        return self.activation(z_new)

    def back_propagate(self,dLdz,dzdy,grad_multiplier=1):
        dLdy = np.matmul(dzdy,dLdz)



        return dLdx

    def update_weights(self):
        """
        Subtract loss gradient from weights and reset
        the rolling loss gradients

        Batch size averages and learning rate have already
        been accounted for in the loss_grads
        """

        self.weights -= self.loss_grads
        self.loss_grads = np.zeros(self.weights.shape)




class FeedForwardNeuralNetwork:
    def __init__(self,
        batch_size = 10,
        epochs = 1000,
        learning_rate = 0.01
        ):

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = []

    def train_datum(self,x,y):

        z_inputs = [None for l in self.layers]
        z_outputs = [None for l in self.layers]
        z_grads = [None for l in self.layers]
        z = x

        # Forward propagate
        for i, l in enumerate(self.layers):
            z_inputs[i] = np.append(z,1)
            z_outputs[i], z_grads[i] = l.propagate(z)
            z = z_outputs[i]

        # Compute loss
        loss, dloss_dy = categorical_cross_entropy_loss(z,y)

        # Back-propagate
        dldz = dloss_dy
        for ind, l in enumerate(reversed(self.layers)):
            ind2 = len(self.layers) - ind - 1

            #print(z_grads[ind2].shape)
            #print(dldz.shape)

            dLdy = np.matmul(z_grads[ind2],dldz)
            dLdw = np.outer(dLdy,z_inputs[ind2])
            l.loss_grads += dLdw * self.learning_rate / self.batch_size
            dldx = np.matmul(np.transpose(l.weights),dLdy)

            # output of next layer is input of previous, but without the 1
            # for the weight
            dldz = dldx[:-1]


    def train_batch(self,X_batch,Y_batch):
        """
        Train on a batch
        """

        for x, y in zip(X_batch, Y_batch):
            self.train_datum(x,y)

        # Update weights on all layers after processing the batch
        for l in self.layers:
            l.update_weights()

    def train(self,X,Y):
        """
        Train on an entire data set
        """

        n_data = X.shape[1]
        n_batches = int(n_data / self.batch_size)

        for epoch in tqdm(range(self.epochs)):
            for batch in range(n_batches):

                X_batch = X[self.batch_size*batch:self.batch_size*(batch+1)]
                Y_batch = Y[self.batch_size*batch:self.batch_size*(batch+1)]
                self.train_batch(X_batch,Y_batch)


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

    nn = FeedForwardNeuralNetwork(
        batch_size = 10,
        epochs = 10,
        learning_rate = 0.01)
    nn.layers = [l1,l2]

    """
    Evaluate a few data points
    """

    nn.train(train_X, train_Y)
    # test_pred_Y = nn.predict(test_X)
    # test_pred_Y = np.round(test_pred_Y)
    # frac_wrong = np.sum(test_pred_Y - test_Y)
    # print("Fraction wrong: {}".format(frac_wrong))


if __name__ == "__main__":
    main()
