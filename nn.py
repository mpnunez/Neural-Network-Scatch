#! ./.venv/bin/python


import numpy as np

from mnist import load_mnist
from loss import categorical_cross_entropy_loss
from activation import softmax, sigmoid
from tqdm import tqdm
import matplotlib.pyplot as plt

class Layer:
    """
    Neural network layer
    """

    def __init__(self,n_input,n_output):

        # Add 1 to input nodes for biases
        self.weights = 10*(np.zeros([n_output,n_input+1]) - 0.5)
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

    def predict_datum(self,x):
        #print(x)
        z = x

        # Forward propagate
        for i, l in enumerate(self.layers):
            z, z_grads = l.propagate(z)

        return z

    def predict(self,X):
        return np.array([self.predict_datum(x) for x in X])

    def confusion_matrix(self,X,Y):
        pred_Y = self.predict(X)

        d = len(Y[0])
        cm = np.zeros((d,d))
        for yp,ya in zip(pred_Y,Y):
            predicted_value = np.argmax(yp)
            actual_value = np.argmax(ya)
            cm[actual_value,predicted_value] += 1
        return cm

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

        return loss


    def train_batch(self,X_batch,Y_batch):
        """
        Train on a batch
        """

        average_loss = 0
        for x, y in zip(X_batch, Y_batch):
            datum_loss = self.train_datum(x,y)
            average_loss += datum_loss / self.batch_size

        # Update weights on all layers after processing the batch
        for l in self.layers:
            l.update_weights()

        return average_loss

    def train(self,X,Y,show_loss_history=True):
        """
        Train on an entire data set
        """

        n_data = len(X)
        n_batches = int(n_data / self.batch_size)

        loss_history = np.zeros([n_batches,self.epochs])

        print("\nStarting training")
        print("{} samples".format(n_data))
        print("{} epochs".format(self.epochs))
        print("{} batches".format(n_batches))

        for epoch in tqdm(range(self.epochs)):
            for batch in tqdm(range(n_batches)):

                X_batch = X[self.batch_size*batch:self.batch_size*(batch+1)]
                Y_batch = Y[self.batch_size*batch:self.batch_size*(batch+1)]
                loss = self.train_batch(X_batch,Y_batch)

                loss_history[batch,epoch] = loss

        if show_loss_history:
            loss_history = np.mean(loss_history,axis=0)
            n_epochs = np.array(range(len(loss_history))) + 1

            plt.plot(n_epochs,loss_history)
            plt.xlabel("Number of epochs")
            plt.ylabel("Loss")
            plt.show()

def train_toy():

    print("Hello World")

    train_X = np.array([[i] for i in range(-10,10)])
    train_Y = np.array([[int(x[0]<=5), int(x[0]>5)] for x in train_X])

    """
    Build neural network architecture
    """

    l1 = Layer(1,1)
    l1.randomize()

    l2 = Layer(1,2)
    l2.randomize()
    l2.activation = softmax

    nn = FeedForwardNeuralNetwork(
        batch_size = len(train_X),
        epochs = 1000,
        learning_rate = 1.0)
    nn.layers = [l1,l2]

    """
    Evaluate a few data points
    """

    nn.train(train_X, train_Y)

def train_mnist():
    print("Hello World")
    (train_X, train_Y), (test_X, test_Y) = load_mnist()

    """
    Build neural network architecture
    """

    n_nodes_first_hidden_layer=10

    l1 = Layer(28*28,n_nodes_first_hidden_layer)
    l1.randomize()

    l2 = Layer(n_nodes_first_hidden_layer,10)
    l2.randomize()
    l2.activation = softmax

    train_X = train_X[::1000]
    train_Y = train_Y[::1000]


    nn = FeedForwardNeuralNetwork(
        batch_size = len(train_X),
        #epochs = 100,
        epochs = 200,
        learning_rate = 0.01)
    nn.layers = [l1,l2]

    cm_pre = nn.confusion_matrix(train_X,train_Y)
    print(cm_pre)

    """
    Evaluate a few data points
    """

    nn.train(train_X, train_Y)
    cm_post = nn.confusion_matrix(train_X,train_Y)
    print(cm_post)

    import pickle
    pickle.dump( nn, open( "nn.p", "wb" ) )



    #train_Y_pred = nn.predict(train_X)


    # test_pred_Y = nn.predict(test_X)
    # test_pred_Y = np.round(test_pred_Y)
    # frac_wrong = np.sum(test_pred_Y - test_Y)
    # print("Fraction wrong: {}".format(frac_wrong))

if __name__ == "__main__":
    train_mnist()
