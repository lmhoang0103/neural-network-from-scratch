from matplotlib.pylab import negative
import numpy as np
import pandas as pd
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layer Class

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
		# Init weights and biases
        # Start small so steps is substantial
        # Weight already in cols instead of rows, so no transpose needed
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Activate Function

# ReLU activation Class
class Activation_ReLU:
	def forward(self, inputs):
		# max(0, input)
		self.output = np.maximum(0, inputs)

# Soft_max activation Class
class Activation_Softmax:
	def forward(self, inputs):
		# exp(x) / sum(exp(x))
		# Subtract max to prevent impact of large value
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities
		
# Loss Function
# Log Loss = -log(pred of correct field) ~ -math.log(y_pred[class])

# If Input have outliers (1 or 0) => skew results => clip values

# Base Loss Class
class Loss:
    # calculate loss from inputs and targets
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_losses = np.mean(sample_losses)  
        return data_losses

# Inherit from Loss Class: Loss Categorical for classification one from multiple
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
		# Number of samples
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # 2 situation, normal and one-hot-encoded
        if len(y_true.shape) ==1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) ==2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods



# Application/Test

# Create dataset
x,y = spiral_data(samples=100, classes=3)

# Create layers/activation/loss
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

# Apply forward pass
dense1.forward(x)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y)
print('loss: ', loss)

