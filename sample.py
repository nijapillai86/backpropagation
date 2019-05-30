import numpy as np 
import random
import math

def processNN(x, y,w1, b1, w2, b):
	print "========= Layer1 ========="

	print w1
	# output = np.dot(x, w) + b
	# print output

	y1 = x[1][0]*w1[0][0] + x[1][1]*w1[1][0] + x[1][2]*w1[2][0] + b1[0]
	y2 = x[1][0]*w1[0][1] + x[1][1]*w1[1][1] + x[1][2]*w1[2][1] + b1[1]
	print y1
	print y2

	print "========= Layer2 output ========="
	
	z = y1*w2[0][0] + y2*w2[1][0] + b
	print z
	# for ele in x:
	# 	print ele[0]

	print "========= error ========="
	error = y[1] - z
	print error[0]
	return error[0],y1,y2



x = np.array([[0, 0, 0],[0, 1, 1],[0, 1, 0]])
y = np.array([0, 3, 2])

#print "========= W1 ========="
w1 = [[0.5, 0.5],[0.5, 0.5],[0.5, 0.5]] 
# print w1

#print "========= bias ========="
b1 =  [[0.5],[0.5]] 
# print b1
#print "========= W2 ========="

w2 = [[0.5],[0.5]] 

b =  0.5

error,y1,y2 = processNN(x,y, w1, b1, w2, b)
ertype = math.isnan(error)
print ertype

while(error > 0.4):
	print "========= Weights changed ========="
	w2[0][0] = 0.5 * error * y1
	w2[1][0] = 0.5 * error * y2
	print w2
	b = 0.5 * error
	b1[0][0] = b1[1][0] = 0.33333 * error
	for i in range(0,3):
		w1[i][0] = 0.33333 * error * w2[0][0] * x[1][i]
		w1[i][1] = 0.33333 * error * w2[1][0] * x[1][i]
		#print w1[i,1]
	#w1 = (1/3)* error * w2[0][0] *np.transpose(inputval) 
	error,y1,y2 = processNN(x,y, w1, b1, w2, b)
	ertype = math.isnan(error)
	print ertype
# for e in error:
# 	print e
