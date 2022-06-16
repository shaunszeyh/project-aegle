import numpy as np
import nnfs

# Ensure the datatypes are correct and a bunch of other overrides
nnfs.init()

class Layer_Dense: # Layer object that initialises and outputs the dot product of input with weights added to biases.
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU: # Activation function object that outputs inputs put into ReLU function
    def forward(self, inputs):
        self.output = np.max(0, inputs)

class Activation_Softmax: # Activation function object for the final layer that outputs inputs put into Softmax function
    def forward(self, inputs):
        exp_values = np.exp(inputs, axis=1, keepdims=True)
        probabilities = exp_values / np.sum(inputs, axis=1, keepdims=True)
        self.output = probabilities





