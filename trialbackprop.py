import numpy as np 
import time
class NeuralNetwork():

	def __init__(self):

		self.W1 = np.array([[0.15, 0.25], [0.20, 0.30]])
		self.W2 = np.array([[0.40, 0.50], [0.45, 0.55]])
		self.b1 = 0.35
		self.b2 = 0.6
		self.learning_rate = 0.1

	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_input, training_output):
		#hidden = np.dot(inputset, weight) + bias
		#print hidden
		output = 1.0
		while training_output !=  output:
		#for x in xrange(1,20):
			print "--------------training_input------------------"
			print training_input
			# Activation function -> sigmoid
			h_out  = self.feedforward(training_input, self.W1, self.b1)
			output = self.feedforward(h_out, self.W2, self.b2)
			print "-------------- hidden output ------------------"
			print h_out
			print "--------------output obtained------------------"
			print output
			error = 0.5 * (training_output - output)**2
			#error = training_output - output
			print error

			#Output layer adjustment
			error_cost = error * self.__sigmoid_derivative(output)
			adjustment = self.learning_rate * np.dot(error_cost, h_out.reshape(1, 2))
			#adjustment = np.dot(h_out.T, error * self.__sigmoid_derivative(output))
			#adjustment = self.learning_rate * h_out.T * error * self.__sigmoid_derivative(output)
			
			#Hidden layer adjustment
			error_prop = h_out * np.dot(error_cost, self.W2.T)
			# print "-----------------------------------------"
			# print error_prop
			error_prop = self.learning_rate * np.dot(training_input.reshape(2, 1), error_prop.reshape(1, 2))
			# print "-----------------------------------------"
			# print error_prop

			if error < 0:
				print "--------------W2------------------"
		        # Adjust the weights.
				self.W2 -= adjustment.reshape(2, 1)

				print self.W2
				self.W1 -= error_prop
				print "--------------- W1 -----------------"
				print self.W1
				self.b1 -= self.learning_rate * error_cost 
				self.b2 -= self.learning_rate * error_cost 
			else :
				print "--------------W2------------------"
		        # Adjust the weights.
				self.W2 += adjustment.reshape(2, 1)

				print self.W2
				self.W1 += error_prop
				print "--------------- W1 -----------------"
				print self.W1
				self.b1 += self.learning_rate * error_cost 
				self.b2 += self.learning_rate * error_cost 
			print "--------------- bias -----------------"
			print self.b1, self.b2		
			time.sleep(0.2)

	def feedforward(self, inputset, weights, bias):
		return self.__sigmoid(np.dot(inputset, weights) + bias)

	# def error_calculation(self, y, y_out):
	# 	output_errors = y - y_out
	# 	print "++++++++++++++++++ output_errors +++++++++++++++++++++++++++"
	# 	print output_errors
	# 	print "+++++++++++++++++++++++++++++++++++++++++++++"

	# 	return output_errors

if __name__ == '__main__':

	# Initialize a single neuron neural network
    neural_network = NeuralNetwork()
    X = np.array([0.05, 0.10])
    y = np.array([0.01, 0.99])
    neural_network.train(X, y)

##########################################################################

    # for epoch in range(200000):  
    # # feedforward
    # zh = np.dot(feature_set, wh)
    # ah = sigmoid(zh)

    # zo = np.dot(ah, wo)
    # ao = sigmoid(zo)

    # # Phase1 =======================

    # error_out = ((1 / 2) * (np.power((ao - labels), 2)))
    # print(error_out.sum())

    # dcost_dao = ao - labels
    # dao_dzo = sigmoid_der(zo) 
    # dzo_dwo = ah

    # dcost_wo = np.dot(dzo_dwo.T, dcost_dao * dao_dzo)

    # # Phase 2 =======================

    # # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    # # dcost_dah = dcost_dzo * dzo_dah
    # dcost_dzo = dcost_dao * dao_dzo
    # dzo_dah = wo
    # dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    # dah_dzh = sigmoid_der(zh) 
    # dzh_dwh = feature_set
    # dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    # # Update Weights ================

    # wh -= lr * dcost_wh
    # wo -= lr * dcost_wo