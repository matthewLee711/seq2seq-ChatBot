#!/usr/bin/python

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
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        # Gradient descent parameters
        self.epsilon = 0.01        # learning rate for gradient descent
        self.reg_lambda = 0.01     # regularization strength

    # Forward propagation
    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        yHat = self.sigmoid(self.z2)
        return yHat

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #  Derivative of sigmoid
    def sigmoidPrime(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)

    # Rate of change of y(hat) with respect to output
    # Compute cost for given X,y, use weights already stored in class.
    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        print(np.ndim(self.yHat))
        print("==============================================================")
        print(np.ndim(self.sigmoidPrime(self.z2)))
        # Backpropagation on layers 2 and 3
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z2))
        dJdW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        # Backpropagation on layers 1 and 2
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z1)
        dJdW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        return dJdW1, dJdW2, db1, db2

    def build_model(self, X, y, nn_hdim, num_passes=20000):
        # Setting Random weights and bias
        self.W1 = np.random.randn(self.inputLayerSize, nn_hdim) / np.sqrt(self.inputLayerSize)
        self.W2 = np.random.randn(nn_hdim, self.outputLayerSize) / np.sqrt(nn_hdim)
        self.b1 = np.zeros((1, nn_hdim))
        self.b2 = np.zeros((1, self.outputLayerSize))
        model = {}

        # Build the nn model
        for i in xrange(0, num_passes):
            # Perform gradientDescent
            dJdW1, dJdW2, db1, db2 = self.costFunctionPrime(X, y)

            # Regularization
            dJdW2 += reg_lambda * W2
            dJdW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dJdW1
            b1 += -epsilon * db1
            W2 += -epsilon * dJdW2
            b2 += -epsilon * db2

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
    # Create dataset
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    createNN = NeuralNetwork()
    model = createNN.build_model(X,y,3,20000)
    print(model)
    # Plot the decision boundary
    plot_decision_boundary(lambda x: createNN.predict(model, x), X, y)

main()
