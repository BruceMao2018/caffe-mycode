#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

my_image = "../nba.png"
my_lable = [0]

image = np.array(ndimage.imread(my_image, flatten=False))
print (image.shape)

image1 = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

print (image1.shape)
print (image1[930039][0])

num_px = image.shape[0]
num_py = image.shape[1]
num_pz = image.shape[2]
print ("x: " + str(num_px) + " y: " + str(num_py) + " z: " + str(num_pz))
image2 = scipy.misc.imresize(image, size=(num_px,num_py)).reshape((num_px*num_py*num_pz,1))
print (image2.shape)
print (image2[930039][0])
