import numpy as np 
import time
class NeuralNetwork():

	def __init__(self):

		self.W1 = np.array([[0.15, 0.25], [0.20, 0.30]])
		self.W2 = np.array([[0.40, 0.50], [0.45, 0.55]])
		self.b = np.array([[0.35], [0.6]])
		# self.b1 = 0.35
		# self.b2 = 0.6
		self.learning_rate = 0.1

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_input, training_output):

		print "--------------training_input------------------"
		print training_input
		output = np.array([0.0, 0.0])
		
		#while(not np.array_equal(output, training_output)) :
		for epoch in range(5): #trains the NN 100 times
			# Activation function -> sigmoid
			print "--------------training_output------------------"
			print training_output
			h_out  = self.feedforward(training_input, self.W1, self.b[0])
			output = self.feedforward(h_out, self.W2, self.b[1])
			print "-------------- hidden output ------------------"
			print h_out
			print "--------------output obtained------------------"
			print output
			error = 0.5 * (training_output - output)**2
			#error = training_output - output
			print "--------------error occurred------------------"
			print error
			print '\n'

			error_tot = 0
			for x in error:
				error_tot += x
			#Output layer adjustment
			error_cost = error_tot * self.__sigmoid_derivative(output)
			# print "------------------- error_cost----------------------"
			# print error_cost
			adjustment = self.learning_rate * np.dot(error_cost, h_out.T)

			#adjustment = np.dot(h_out.T, error * self.__sigmoid_derivative(output))
			#adjustment = self.learning_rate * h_out.T * error * self.__sigmoid_derivative(output)
			# print "------------------- adjustment----------------------"
			# print adjustment
			
			#Hidden layer adjustment
			error_prop = h_out * np.dot(error_cost, self.W2.T)
			# print "-----------------------------------------"
			#print error_prop
			error_prop = self.learning_rate * np.dot(training_input, error_prop.T)
			# print "------------------error_prop-----------------------"
			# print error_prop

			# Adjust the weights.
			for i in range(len(output)):
				j = 0
				#print training_output[i], error[i]
				if training_output[i] > output[i]:
					print "++++++++++++ training_output > error ++++++++++++ "
					self.W2[j][i] += adjustment
					self.W2[j+1][i] += adjustment
					self.W1[j][i] += error_prop
					self.W1[j+1][i] += error_prop
					self.b[i] += self.learning_rate * error_tot 

				elif training_output[i] < output[i] :
					print "++++++++++++ training_output < error ++++++++++++ "
					#print "--------------W2------------------"
			        # Adjust the weights.
					self.W2[j][i]   -= adjustment
					self.W2[j+1][i] -= adjustment
					self.W1[j][i]   -= error_prop
					self.W1[j+1][i] -= error_prop

					self.b[i] -= self.learning_rate * error_tot 

			print "--------------W2------------------"

			print self.W2

			print "--------------- W1 -----------------"

			print self.W1

			print "--------------- bias -----------------"
			print self.b	
			time.sleep(0.2)

			print '\n'
			print '\n'
			print '\n'


	def feedforward(self, inputset, weights, bias):
		return self.__sigmoid(np.dot(inputset, weights) + bias)

if __name__ == '__main__':

	# Initialize a single neuron neural network
    neural_network = NeuralNetwork()
    X = np.array([0.05, 0.10])
    y = np.array([0.01, 0.99])
    neural_network.train(X, y)

##########################################################################