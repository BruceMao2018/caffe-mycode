#coding=utf-8
import numpy as np

x = np.array([[2, 4, 10],
	      [3, 6, 9]])

print ("x: " + str(x))
print ("sqrt: " + str(np.sqrt(x)))
print ("square: " + str(np.square(x)))

W2 = np.random.random((10, 8))
print (W2)
"""


W0 =  np.random.randn(2, 4)
W1 =  np.random.randn(2, 4) / np.sqrt(2)
print (W0)
print (W1)
print (np.sum(W1))
