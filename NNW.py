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
        # Weight already in cols instead of rows, so no transpose needed
		self.weights = np.sqrt(2.0 / n_inputs) * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.inputs = inputs  # Save inputs for backpropagation
		self.output = np.dot(inputs, self.weights) + self.biases
	
	# Backward pass
	# dvalues = dL/doutputs
	# dL/dW = XT . dL/doutputs
	# dL/db = sum of dL/doutputs 
	# dL/dX = dL/doutputs . WT
	def backward(self, dvalues):
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

# Activate Function

# ReLU activation Class
class Activation_ReLU:
	def forward(self, inputs):
		# max(0, input)
		self.input = inputs
		self.output = np.maximum(0, inputs)
		
	# Backward Pass
	# Want d(Relu(input))/d(input) = 1 if  input > 0
	# In otherw word, in regard to d(Loss), it does NOT change the chain of derivative if dinputs >0,
	# if <=0 => remove the WHOLE chain
	# dvalues = dL/dvalues = dL/d(ReLU(inputs))
	# dinputs = dL/dinputs = dL/dvalues * dvalues/d(inputs) = dvalues if inputs >0 else 0
	def backward(self, dvalues):
		self.dinputs = dvalues.copy()
		self.dinputs[self.inputs <=0] =0

# Soft_max activation Class
# Have special derivative if combined with Categorical Cross-Entropy Loss
class Activation_Softmax:
	def forward(self, inputs):
		# exp(x) / sum(exp(x))
		# Subtract max to prevent impact of large value
		self.inputs = inputs
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities
	
	def backward(self, dvalues):
		# Create uninitialized array
		self.dinputs = np.empty_like(dvalues)
		# Iterate over outputs and gradients
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			# Flatten output array
			single_output = single_output.reshape(-1,1)
			# Calculate Jacobian matrix of the softmax function
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
			# Calculate sample-wise gradient
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

		
# Loss Function
# Log Loss = -log(pred of correct field) ~ -math.log(y_pred[class])

# If Input have outliers (1 or 0) => skew results => clip values

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

# Inherit from Loss Class
class Loss_CategoricalCrossentropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		# 2 situation, normal and one-hot-encoded
		if len(y_true.shape) ==1:
			correct_confidences = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) ==2:
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods
	
	# dvalue = Loss(model output)
	# Want to find d(Loss)/d(model output) = -[y_true]/[model output] (For derivative of log(x) = 1/x)
	def backward(self, dvalues, y_true):
		# Number of samples = batch sizes
		samples = len(dvalues)
		labels = len(dvalues[0])

		# If y_true is an array of labels => convert to matrix using one-hot-encoding
		if len(y_true.shape) ==1:
			y_true = np.eye(labels)[y_true]
		
		self.dinputs = -y_true / dvalues
		# Normalize gradient, in calculate loss already took mean, so here also need to take mean
		self.dinputs = self.dinputs / samples


# Combine soft max + categorialCrossentropy derivative
# Simple derivative when combine: Pred - True
class Activation_Softmax_Loss_CategoricalCrossentropy():
	def __init__(self):
		self.activation = Activation_Softmax()
		self.loss = Loss_CategoricalCrossentropy()

	def forward(self, inputs, y_true):
		self.activation.forward(inputs)
		self.output = self.activation.output
		# Calculate and return loss value
		return self.loss.calculate(self.output, y_true)
	# Backward pass
	def backward(self, dvalues, y_true):
		# Number of samples
		samples = len(dvalues)
		# If y_true is one-hot-encoded, convert to labels
		if len(y_true.shape) ==2:
			y_true = np.argmax(y_true, axis=1)
		# Copy so we can safely modify
		self.dinputs = dvalues.copy()
		# Calculate gradient
		# dinputs = np.arrange(samples) - y_true
		self.dinputs[range(samples), y_true] -=1
		# Normalize gradient
		self.dinputs = self.dinputs / samples

# Optimizer: SGD with momentum
class Optimizer_SGD:
	def __init__(self, learning_rate = 1):
		self.learning_rate = learning_rate
	
	def update_params(self, layer):
		layer.weights -= self.learning_rate * layer.dweights
		layer.biases -= self.learning_rate * layer.dbiases

# Apply Model codes
# Rerun and update after class changes

# Create dataset
x,y = spiral_data(samples=100, classes=3)
# Create Layers
dense1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(5,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=100)

# Train in epochs
for epoch in range(10001):
	# Forward pass
	dense1.forward(x)
	activation1.forward(dense1.output)
	dense2.forward(activation1.output)
	loss = loss_activation.forward(dense2.output, y)

	# Calculate accuracy
	predictions = np.argmax(loss_activation.output, axis=1)
	accuracy = np.mean(predictions == y)

	if epoch % 100 ==0:
		print(f'Epoch: {epoch}, ' +
			  f'loss: {loss:.3f}, ' +
			  f'accuracy: {accuracy:.3f}')
	
	# Backward pass
	loss_activation.backward(loss_activation.output, y)
	dense2.backward(loss_activation.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	# Update weights and biases
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
