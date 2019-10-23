#coding=utf-8
from dnn import *
from lr_utils import *
import numpy as np

##############################################################################################################################################################
"""
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.0
test_x = test_x_flatten / 255.0


### CONSTANTS ###
layers_dims = (12288, 20, 7, 5, 1) #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2000, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
#print (train_y)
#print (test_y)

#### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 1000, print_cost=True)
#predictions_train = predict(train_x, train_y, parameters)
#predictions_test = predict(test_x, test_y, parameters)
"""



X = np.array([[1, 1, 0, 1, 1],
	      [0, 1, 9, 2, 0],
	      [3, 2, 1, 2, 1]])
Y = np.array([[0, 1, 0, 1, 0]])

#paras = initialize_parameters_deep(layers_dims)

#AL,caches = L_model_forward(X,paras)
#print (AL)

#grads = L_model_backward(AL,Y,caches)
#print (grads)

layers_dims = (3, 4, 5, 2, 1)

#parameters = two_layer_model(X, Y, layers_dims, num_iterations = 2500, print_cost=True,isPlot=False)
parameters = L_layer_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=2000, print_cost=True,isPlot=False)
p = predict(X, Y, parameters)
#print (p)
