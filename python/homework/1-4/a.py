# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 15:21:43 2018

@author: hp
"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
#from testCases_v2 import linear_forward_test_case,linear_activation_forward_test_case,L_model_forward_test_case,linear_backward_test_case,linear_activation_backward_test_case,L_model_backward_test_case,update_parameters_test_case
from dnn_utils_v2 import sigmoid,sigmoid_backward,relu,relu_backward
import scipy
from PIL import Image
from scipy import ndimage

plt.rcParams['figure.figsize']=(5.0,4.0)#set default size of plots
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

np.random.seed(1)

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    assert(W1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))
    
    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}
    return parameters

#parameters = initialize_parameters(3,2,1)
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))

def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters={}
    L=len(layer_dims) # number of layers in the network
    
    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])/ np.sqrt(layer_dims[l-1])
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters

#parameters = initialize_parameters_deep([5,4,3])
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))

def linear_forward(A,W,b):
    Z=np.dot(W,A)+b
    assert(Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)
    
    return Z,cache


#A, W, b = linear_forward_test_case()
#Z, linear_cache = linear_forward(A, W, b)
#print("Z = " + str(Z))


def linear_activation_forward(A_prev,W,b,activation):
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)
        
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(linear_cache,activation_cache)
    
    return A,cache

#A_prev, W, b = linear_activation_forward_test_case()
#
#A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
#print("With sigmoid: A = " + str(A))
#
#A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
#print("With ReLU: A = " + str(A))
    
def L_model_forward(X,parameters):
    caches=[]
    A=X
    L=len(parameters)//2 # number of layers in the neural network
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b' + str(l)],"relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

#X, parameters = L_model_forward_test_case()
#AL, caches = L_model_forward(X, parameters)
#print("AL = " + str(AL))
#print("Length of caches list = " + str(len(caches)))
    
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost

def linear_backward(dZ,cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    
    assert(dA_prev.shape==A_prev.shape)
    assert(dW.shape==W.shape)
    assert(db.shape==b.shape)
    
    return dA_prev,dW,db

## Set up some test inputs
#dZ, linear_cache = linear_backward_test_case()
#
#dA_prev, dW, db = linear_backward(dZ, linear_cache)
#print ("dA_prev = "+ str(dA_prev))
#print ("dW = " + str(dW))
#print ("db = " + str(db))

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev,dW,db

#AL, linear_activation_cache = linear_activation_backward_test_case()
#
#dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
#print ("sigmoid:")
#print ("dA_prev = "+ str(dA_prev))
#print ("dW = " + str(dW))
#print ("db = " + str(db) + "\n")
#
#dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
#print ("relu:")
#print ("dA_prev = "+ str(dA_prev))
#print ("dW = " + str(dW))
#print ("db = " + str(db))

def L_model_backward(AL,Y,caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

#AL, Y_assess, caches = L_model_backward_test_case()
#grads = L_model_backward(AL, Y_assess, caches)
#print_grads(grads)
    
def update_parameters(parameters,grads,learning_rate):
    L=len(parameters)//2
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

#parameters, grads = update_parameters_test_case()
#parameters = update_parameters(parameters, grads, 0.1)
#
#print ("W1 = "+ str(parameters["W1"]))
#print ("b1 = "+ str(parameters["b1"]))
#print ("W2 = "+ str(parameters["W2"]))
#print ("b2 = "+ str(parameters["b2"]))
    

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)


    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    print("Accuracy: "  + str(np.sum((p == y)/m)))

    return p


################################################################################33###############################################################
def load_data():
    train_dataset = h5py.File('../1-2/logistic/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('../1-2/logistic/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# GRADED FUNCTION: two_layer_model

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    #m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###

    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
	A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
	A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

        # Compute cost
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(A2, Y)
        ### END CODE HERE ###


        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        ### END CODE HERE ###

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2


        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training example
        if print_cost and i % 200 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 200 == 0:
            costs.append(cost)

    """
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    """

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost

    # Parameters initialization.
    ### START CODE HERE ###
    parameters = initialize_parameters_deep(layers_dims)
    ### END CODE HERE ###

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###

        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###

        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###

        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Print the cost every 100 training example
        if print_cost and i % 200 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 200 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

##############################################################################################################################################################
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
## Example of a picture
#index = 10
#plt.imshow(train_x_orig[index])
#print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Explore your dataset 
#m_train = train_x_orig.shape[0]
#num_px = train_x_orig.shape[1]
#m_test = test_x_orig.shape[0]

#print ("Number of training examples: " + str(m_train))
#print ("Number of testing examples: " + str(m_test))
#print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
#print ("train_x_orig shape: " + str(train_x_orig.shape))
#print ("train_y shape: " + str(train_y.shape))
#print ("test_x_orig shape: " + str(test_x_orig.shape))
#print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

#print ("train_x's shape: " + str(train_x.shape))
#print ("test_x's shape: " + str(test_x.shape))





#### CONSTANTS DEFINING THE MODEL ####
#n_x = 12288     # num_px * num_px * 3
#n_h = 7
#n_y = 1
#layers_dims = (n_x, n_h, n_y)

#parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 3000, print_cost=True)
#predictions_train = predict(train_x, train_y, parameters)
#predictions_test = predict(test_x, test_y, parameters)






### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
