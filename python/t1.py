import numpy as np
#A = np.random.randn(4,3)
A = np.array([[1,1,1], [2,2,2], [3,3,3], [4,4,4]])
B = np.sum(A, axis = 0, keepdims = True)

print (A)
print (B)
