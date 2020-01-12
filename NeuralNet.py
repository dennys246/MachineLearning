import numpy as np


class NeuralNetwork():

	def __init__(self):
		np.random.seed(1)

		self.weights = 2 * np.random.random((3,1)) - 1

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, train_inputs, train_outputs, train_iterations):
		for iteration in range(train_iterations):

			output = self.think(train_inputs)

			error = train_outputs - output

			adjustments = np.dot(train_inputs.T, error * self.sigmoid_derivative(output))

			self.weights += adjustments

	def think(self, inputs):

		inputs = inputs.astype(float)

		output = self.sigmoid(np.dot(inputs, self.weights))

		return output


if __name__ == '__main__':
	NN = NeuralNetwork()

	print("Weights:")
	print(NN.weights)

	train_inputs = np.array([[1,0,1],
					 [0,1,0],
					 [1,1,0],
					 [0,1,1]])

	train_outputs = np.array([[1,0,1,0]]).T

	print('How many iterations would you like to run?')
	iterations = int(input())

	NN.train(train_inputs, train_outputs, iterations)

	print("Weights:")
	print(NN.weights)

	A = str(input("Input 1: "))
	B = str(input("Input 2: "))
	C = str(input("Input 3: "))

	print("New scenario data:", A , B, C)
	print("Outputting data...")
	print(NN.think(np.array([A,B,C])))
