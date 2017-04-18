import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

class NeuralNetwork():
    def __init__(self):
        # Define dimensionality of NN
        self.nn_input_dim = 2      # input layer dimensionality
        self.nn_output_dim = 2     # output layer dimensionality
        self.hiddenLayerSize = 10
        # Gradient descent parameters
        self.epsilon = 0.01        # learning rate for gradient descent
        self.reg_lambda = 0.01     # regularization strength

    def forward(self, X):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidPrime(self, x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    # Helper function to predict an output (0 or 1)
    def predict(self, model, x):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def build_model(self, X, y, nn_hdim, num_passes=20000, print_loss=False):

        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        W1 = np.random.randn(self.nn_input_dim, nn_hdim) / np.sqrt(self.nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, self.nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, self.nn_output_dim))

        num_examples = len(X) # training set size
        # This is what we return at the end
        model = {}

        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):
            # probs = forward(W1, b1, W2, b2)
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(num_examples), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2

            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        return model

def plot_decision_boundary(pred_func, X, y):

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.title("Decision Boundary for hidden layer size 3")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def main():
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    createNN = NeuralNetwork()
    model = createNN.build_model(X,y,3,20000)
    # Plot the decision boundary
    plot_decision_boundary(lambda x: createNN.predict(model, x), X, y)


main()
