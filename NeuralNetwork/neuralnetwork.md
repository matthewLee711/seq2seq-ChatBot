# initialization
Setting input nodes
Setting output nodes
Choosing dimensionality

# forward propagation
f(W,x) = Wx + b

essentially, taking the activation function of the summation of weights x inputs. This will give a probability weight of the neuron.
input - nodes that have input. 1 is data, 2 is number of input nodes
weights - values given to synapses (keep adding rows to continually add more data to f - propagate on)
dot product the inputs and weights. 2 is inputs, 3 is the number of hidden nodes.

input: 1x2        weights: 2 x 3
[x1,x2]           [w1, w2, w3]
                  [w4, w5, w6]

yhat = sigmoid(summation(wx + b))

# gradient descent
Taking the total predicted weight probabilities and subtracting it from the fitted probabilities.
The ultimate goal of gradient descent is to get the minima for the loss function.
One issue when finding the minima is gradient descent may accidentally consider the local minima to be the minima of the total loss function (Non-convex). To counter this, use stochastic gradient descent or mini batch gradient descent. This will take

summation(0.5(y - yHat)^2)

Numerical (gradient) estimation - take the minimum estimation by testing left and right. It works well, but there is faster
# gradient descent
To quanitfy how wrong our predictions are, we use a cost function ( summation(0.5(y - yHat)^2) ). When you train a NN, this means you are minimizing a cost function. Cost is a function of two things: our inputs (examples) and weights on synapses. Since we dont have control over data, minimize the cost function by changing weights.
J = sum(.5 * (y - f(f(XW)W))) <-- we take the partial derivative of this function because we need to evaluate each individual weight. We want to find which way is downhill or the rate of change of J to weights. SO if DJ/DW is positive, then change is uphill, we need negative.
The reason we use the sum of squared errors is to prevent non-convex errors in gradient descent (creates large parabola?).

We need to calculate two separate partial derivatives. One for W1 and W2.

Dj = D sum(.5 * (y - yHat)^2)         D .5 * (y - yHat)^2
--   ------------------------  ---->  -------------------
Dw              Dw                            Dw

sum rule in differentiation

# Entire NN equation

sum(.5 * (y - sum(sig(sig(XW)W)))^2)

# Cost function - take partial derivative bc evaluate each weight
sum(.5 * (y - f(f(XW)W)))
