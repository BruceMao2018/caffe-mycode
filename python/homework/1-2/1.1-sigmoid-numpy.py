#coding=utf-8
#Exercise: Implement the sigmoid function using numpy.

#Instructions: x could now be either a real number, a vector, or a matrix. The data structures we use in numpy to represent these shapes (vectors, matrices…) are called numpy arrays. You don’t need to know more for now. 

import numpy as np

def sigmoid(x):
	return 1/(1+np.exp(x))

s = sigmoid(3)
print (s)

x = [1, 2, 3]
s = sigmoid(x)
print (s)

x = np.array([1, 2, 3])
s = sigmoid(x)
print (s)

x = np.array([[1, 2, 3]])
s = sigmoid(x)
print (s)
