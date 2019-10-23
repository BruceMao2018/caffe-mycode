#coding=utf-8
import numpy as np
import sys
sys.path.append('../') #将上一级目录加入到系统目录，否则下面的bruce将无法找到
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import bruce as bruce

def load_dataset(is_plot = True):
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    if is_plot:
        plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    return train_X, train_Y

train_X, train_Y = load_dataset(is_plot=True)
print ("train_X: " + str(train_X.shape))
print ("train_Y: " + str(train_Y.shape))

layers_dims = (train_X.shape[0], 5, 2, 1)
#parameters = bruce.L_layer_model_Reg(train_X, train_Y, layers_dims, learning_rate=0.0007, num_iterations=10000, print_cost=True, initialization="he", isPlot=False, lambd=0, keep_prob=1)
parameters =  bruce.model_Reg_prob_gradCheck_momentun_RMS_Adam_MiniBatch(train_X, train_Y, layers_dims, learning_rate=0.0007, num_iterations=10000, print_cost=True, initialization="he", isPlot=False, lambd=0, keep_prob=1, beta1=0.9, beta2=0.999, epsilon=1e-8, mini_batch_size=64, optimizer="adam")

print ("训练集:")
bruce.predict(train_X, train_Y, parameters)

"""
X = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
	          [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
	          [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
	          [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

Y = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]]) #当第二及第三个都为1时，则Y为1，其他情况为0
train_X = X
train_Y = Y
layers_dims = (train_X.shape[0], 20, 3, 1)

print ("X.shape: " + str(train_X.shape))
print ("Y.shape: " + str(train_Y.shape))

parameters = bruce.L_layer_model_Reg(train_X, train_Y, layers_dims, learning_rate=0.01, num_iterations=10000, print_cost=True, initialization="he", isPlot=False, lambd=0, keep_prob=1)

print ("训练集:")
bruce.predict(train_X, train_Y, parameters)
"""
