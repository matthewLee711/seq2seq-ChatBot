# seq2seq-ChatBot
Sequence to sequence learning based chatbot

The ultimate goal of this repository is to experiment with the training and creation of a recurrent neural network based chatbot. 
In the NeuralNetwork folder, there is example code of an artificial neural network which it's purpose is to help me grasp the concept
of neural networks. Within the "notebooks" folder there are two python notebooks. The backpropagation notebook contains the derivation 
of gradient descent with a sigmoid function while the neural network notebook shows an example of training a neural over a dataset.

The NeuralNetworkImpl is a direct implementation of a neural network. One uses a tan(h) activation function while the other uses a 
sigmoid activation function. Switching the activation function requires a lot of changes to the codebase because you have to find the 
derivative of the desired function and replace the code.
