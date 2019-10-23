# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 19:04:47 2018

@author: hp
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy 
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from model import models
from model import predict

#loading the data(cat/non-cat)
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()

##Example of a picture
#index=25
#plt.imshow(train_set_x_orig[index])
#print("y="+str(train_set_y[:,index])+",it's a '"+classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"'picture.")

#m_train.m_test,num_px
m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shape[0]
num_px=train_set_x_orig.shape[1]

#print("Number of training examples:m_train="+str(m_train))
#print("Number of test examples:m_test="+str(m_test))
#print ("Height/Width of each image: num_px = " + str(num_px))
#print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
#print ("train_set_x shape: " + str(train_set_x_orig.shape))
#print ("train_set_y shape: " + str(train_set_y.shape))
#print ("test_set_x shape: " + str(test_set_x_orig.shape))
#print ("test_set_y shape: " + str(test_set_y.shape))

#reshape the training and test data
train_set_x_flatten=train_set_x_orig.reshape(m_train,-1).T
test_set_x_flatten=test_set_x_orig.reshape(m_test,-1).T

#print("train_set_x_flatten shape"+str(train_set_x_flatten.shape))
#print ("train_set_y shape: " + str(train_set_y.shape))
#print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
#print ("test_set_y shape: " + str(test_set_y.shape))
#print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))#合理性检查

#center and standardize the dataset
train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

d=models(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate = 0.005, print_cost = True)

## Example of a picture that was wrongly classified.
#index = 1
#plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
#print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")

##plot learning curve
#costs=np.squeeze(d['costs'])
#plt.plot(costs)
#plt.ylabel('cost')
#plt.xlabel('iterations (per hundreds)')
#plt.title("Learning rate =" + str(d["learning_rate"]))
#plt.show()

##choice of learning rate
#learning_rates=[0.01,0.001,0.0001]
#model={}
#for i in learning_rates:
#    print("learning rate is:"+str(i))
#    model[str(i)] = models(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
#    print ('\n' + "-------------------------------------------------------" + '\n')
#
#for i in learning_rates:
#    plt.plot(np.squeeze(model[str(i)]["costs"]), label= str(model[str(i)]["learning_rate"]))
#
#plt.ylabel('cost')
#plt.xlabel('iterations')
#
#legend = plt.legend(loc='upper center', shadow=True)
#frame = legend.get_frame()
#frame.set_facecolor('0.90')
#plt.show()

## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "123.png"   # change this to the name of your image file 
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")