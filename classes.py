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

class Loss: # Runs and returns output of loss value function
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy: # Calculates loss value with Categorical Cross Entropy
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if (len(y_true) == 1):
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif (len(y_true) == 2):
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods