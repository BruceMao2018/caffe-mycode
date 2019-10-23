#coding=utf-8
import numpy as np
import bruce
import gradient_check

X = np.array([[0, 0, 1, 0, 1, 1], 
	          [0, 0, 1, 1, 0, 1], 
	          [0, 1, 1, 1, 1, 1]]) 

Y = np.array([[0, 0, 1, 0, 0, 1]])
train_X = X
train_Y = Y
layers_dims = (train_X.shape[0], 5, 3, 1)
parameters = bruce.L_layer_model_Reg(train_X, train_Y, layers_dims, learning_rate=0.01, num_iterations=500, print_cost=True, initialization="he", isPlot=False, lambd=0, keep_prob=1)
#print ("parameters: " + str(parameters))
#vec = gradient_check.dictionary_to_vector(parameters)
#print ("vec: " + str(vec))
#print ("vec to para: " + str(gradient_check.vector_to_dictionary(vec, layers_dims)))
