#coding=utf-8
import numpy as np

x = np.array([[10, 11, 240, 100, 121],
 [80, 11, 190, 255, 9],
 [90, 101, 120, 0, 9]])

print ("x/255:\n" + str(x/255))
print ("x/255.0:\n" + str(x/255.0))
