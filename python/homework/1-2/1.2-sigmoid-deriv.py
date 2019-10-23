#coding=utf-8
#1.2 - Sigmoid gradient
#Exercise: Implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. 
#The formula is: sigmoid_derivative(x)=σ′(x)=σ(x)(1−σ(x))

import numpy as np

def  sigmoid(x):
	return 1/(1 + np.exp(x)) #sigmoid(x)的运算公式为：f(x) = 1 / (1 + exp(x))

def sigmoid_der(x):
	return sigmoid(x) * (1 -sigmoid(x)) #f(x)对x求导的运算公式为: f'(x) = f(x) * (1 - f(x))

print (sigmoid_der(3))

x = [1, 2, 3]
print (sigmoid_der(x))
