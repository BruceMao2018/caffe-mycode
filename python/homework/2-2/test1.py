#coding=utf-8
import numpy as np
import math

np.random.seed(1)
"""
X = np.array([[1,2,3,4,5,6,7,8,9],
				[9,8,7,6,5,4,3,2,1],
				[1,2,3,4,5,6,7,8,9]])
print (X)
m = X.shape[1]
permutation = list(np.random.permutation(m))
print ("per: " + str(permutation))
shuffled_X = X[:,permutation]   #将每一列的数据按permutation的顺序来重新排列。
print ("shuffled_X: " + str(shuffled_X))
p2 = [1, 2, 3, 4, 5, 6, 7]
shuffled_X2 = X[:,p2]   #将每一列的数据按permutation的顺序来重新排列。
print ("shuffled_X2: " + str(shuffled_X2))

mini_batch_X = shuffled_X[:, 2:5]
print (mini_batch_X)
"""
m = 9
b = 2
y1 = m/b
y2 = math.floor(m/b)
print (y1)
print (y2)
