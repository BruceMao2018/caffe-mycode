#coding=utf-8
# Exercise: Build a function that returns the sigmoid of a real number x. Use math.exp(x) for the exponential function.

# Reminder: 
# sigmoid(x)=1/(1+e(-x)) is sometimes also known as the logistic function. It is a non-linear function used not only 
# in Machine Learning (Logistic Regression), but also in Deep Learning.

import numpy as np
import math

def sigmoid(x):
	return 1.0/(1+1/math.exp(x))

print (sigmoid(3))

x = [1,2,3]
print (np.exp(x))

#sigmoid(x)
# you will see this give an error when you run it, because x is a vector.
#In fact, if x=(x1,x2,...,xn)x=(x1,x2,...,xn) is a row vector then np.exp(x)np.exp(x) will apply the exponential function to every element of x. The output will thus be: np.exp(x)=(ex1,ex2,...,exn)

x = np.array([1, 2, 3])
print (np.exp(x)) # result is (exp(1), exp(2), exp(3))

#Furthermore, if x is a vector, then a Python operation such as s=x+3s=x+3 or s=1xs=1x will output s as a vector of the same size as x.
x = x + 3
print (x)
