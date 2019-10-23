#coding=utf-8
import numpy as np

#指定随机种子，以每次随机值都一样，便于测试
#np.random.seed(1)

#4.1 - Helper functions

def sigmoid(x, Deriv = False):
	if( Deriv == False): #前向传播
		return 1/(1+np.exp(-x))
	return x * (1 -x) #反向传播

#4.2 - Initializing parameters
#Exercise: Implement parameter initialization in the cell below. You have to initialize w as a vector of zeros. If you don’t know what numpy function to use, look up np.zeros() in the Numpy library’s documentation.

"""
this function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
"""
def init_zeros(dim):
	w = np.zeros((dim, 1))
	b = 0
	assert(w.shape == (dim, 1))
	assert(isinstance(b, float) or isinstance(b, int))
	return w, b

def init_rand(dim):
	w = np.random.random((dim, 1))
	b = 0
	return w, b

def init_rand_normalize(dim):
	w = np.random.random((dim, 1)) #获得0-1之间的随机值
	w = w * 2 - 1 #返回的矩阵值，介于-1到1之间
	return w, b

dim = 2
w, b = init_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))

w, b = init_rand(dim)
print ("w = " + str(w))
print ("b = " + str(b))

w, b = init_rand_normalize(dim)
print ("w = " + str(w))
print ("b = " + str(b))
