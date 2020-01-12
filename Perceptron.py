import numpy as np

# Normalizing/Activation function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# Error weighted derivative function for backpropagation
def sigmoid_derivative(x):
	return x * (1 - x)

# Define training input
train_inputs = np.array([[1,0,0],
						 [0,0,1],
						 [1,1,0],
						 [0,1,1]])

# Define training outputs - transpose row
train_outputs = np.array([[1,0,1,0]]).T

# Seed random numbers
np.random.seed(1)

weights = 2 * np.random.random((3,1)) - 1

print('How many iterations would you like to run?')
iteration = int(input())

for iteration in range(iteration):

	input_layer = train_inputs

	outputs = sigmoid(np.dot(input_layer, weights))

	error = train_outputs - outputs

	adjustment = error * sigmoid_derivative(outputs)

	weights += np.dot(input_layer.T, adjustment)

print (outputs)