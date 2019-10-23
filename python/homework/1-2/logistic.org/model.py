# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 18:32:14 2018

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy 
from PIL import Image
from scipy import ndimage

def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s
#print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b    

#dim = 3
#w,b=initialize_with_zeros(dim)
#print("w="+str(w))
#print("b="+str(b))

#前向和反向传播
def propagate(w,b,X,Y):
    m=X.shape[1]
    #fotward propagation
    A=sigmoid(np.dot(w.T,X)+b)
    cost = -(1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    #backward propagation
    dw=(1.0/m)*np.dot(X,(A-Y).T)
    db=(1.0/m)*np.sum(A-Y)
    
    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost = np.squeeze(cost)#去掉cost中只有一行或一列的维度，eg：（3,1,2）变成（3,2）
    assert(cost.shape==())
    
    grads={"dw":dw,
           "db":db}
    return grads,cost

#w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
#grads, cost = propagate(w, b, X, Y)
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))
#print ("cost = " + str(cost))

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]   
        w=w-learning_rate*dw
        b=b-learning_rate*db
        
        #record the costs
        if i%100 == 0:
            costs.append(cost)
            
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))    
        
    params ={"w":w,
             "b":b}
    grads={"dw":dw,
           "db":db}
    return params, grads, costs

#w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
#params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
#
#print ("w = " + str(params["w"]))
#print ("b = " + str(params["b"]))
#print ("dw = " + str(grads["dw"]))
#print ("db = " + str(grads["db"]))

def predict(w,b,X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
            
    assert(Y_prediction.shape==(1,m))
    return Y_prediction

#w = np.array([[0.1124579],[0.23106775]])
#b = -0.3
#X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
#print ("predictions = " + str(predict(w, b, X)))

def models(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate = 0.5, print_cost = False):
    #initialize_with_zeros
    w,b=initialize_with_zeros(X_train.shape[0])
    #计算grads,cost
    grads,cost=propagate(w,b,X_train,Y_train)
    #梯度下降
    params, grads, costs=optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w=params["w"]
    b=params["b"]
    #预测
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)    
    
    #输出正确率
    print("train accuracy:{} %".format(100 - np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy:{} %".format(100 - np.mean(np.abs(Y_prediction_test-Y_test))*100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}    
    return d
