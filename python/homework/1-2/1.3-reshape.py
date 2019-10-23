#coding=utf-8
#Two common numpy functions used in deep learning are np.shape and np.reshape(). 
#- X.shape is used to get the shape (dimension) of a matrix/vector X. 
#- X.reshape(…) is used to reshape X into some other dimension.

#For example, in computer science, an image is represented by a 3D array of shape (length,height,depth=3)(length,height,depth=3). However, when you read an image as the input of an algorithm you convert it to a vector of shape (length∗height∗3,1)(length∗height∗3,1). In other words, you “unroll”, or reshape, the 3D array into a 1D vector.

import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print (x.shape)
print (x)

x = x.reshape(3,2)
print (x.shape)
print (x)

x = x.reshape(6,1)
print (x.shape)
print (x)

#Exercise: Implement image2vector() that takes an input of shape (length, height, 3) and returns a vector of shape (length*height*3, 1). For example, if you would like to reshape an array v of shape (a, b, c) into a vector of shape (a*b,c) you would do:

#v = v.reshape((v.shape[0]*v.shape[1], v.shape[2])) #v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
#Please don’t hardcode the dimensions of image as a constant. Instead look up the quantities you need with image.shape[0], etc.

def image2vec(image):
	v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
	return v

img1 = np.random.rand(32, 32, 3)
print (img1.shape)

vec1 = image2vec(img1)
print (vec1.shape)
