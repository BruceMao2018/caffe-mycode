#coding=utf-8
#	试验目的:验证简易的前向，反向传播
#	Bruce.Mao
#	Version 01	2017.12.22
#	Version 02	2019.3.15
#	5个输入样本L0，每个样本3个特征，及输入层X为(5,3)的矩阵
#
#	L1层有4个输出，即4个特征，因为有5个样本，所以L1层的数据为（5,4）的矩阵
#	由X*w0 = L0, (5,3) * w0 = (5,4)可推导出w0为（3,4）的矩阵
#	
#	L2 层为输出层，此程序为二分类程序，故5个样本的输出值，要么是0，要么是1，因此可得出L2的值为（5,1）的矩阵
#	由L1*w1 = (5,4)*w1 = (5,1), 可推导出w1为（4,1）的矩阵
#
#
#		L0 -w0	L1 -w1	L2(output)
#		___		___
#		| |	 _ 	| |
#	x1	|_|	| |	|_|S
#	 	| |	|_|S	| |i
#	x2	|_|	| |i	|_|g
#	 	| |	|_|g	| |m
#	x3	|_|	| |m	|_|o
#	 	| |	|_|o	| |i
#	x4	|_|	| |i	|_|d
#	 	| |	|_|d	| |
#	x5	|_|	   	|_|
#
#
#	x	|_|	|_|	|_|

import numpy as np
def sigmoid(x, deriv = False):
	if( deriv == True): #反向传播
		return x * (1 - x)
	return 1/(x+np.exp(-x)) #前向传播

np.random.seed(1) #设定一个固定的随机因子，以便每次测试的时候得到一个相同的值

x = np.array([[0,0,1],
				[0,1,1],
				[1,0,1],
				[1,1,1],
				[0,0,1]])
y = np.array([[0],
			 [0],
			 [0],
			 [1],
			 [0]])

w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1

print (y.shape)
print (w0)
print (w1)


for j in xrange(5000):
	Z1 = np.dot(x, w0)
	A1 = sigmoid(Z1)

	Z2 = np.dot(A1, w1)
	A2 = sigmoid(Z2)

	dA2 = y - A2
	dZ2 = dA2 * sigmoid(A2, True)
	dA1 = np.dot(dZ2, w1.T) 
	dZ1 = dA1 * sigmoid(A1, True)

	dW1 = np.dot(A1.T, dZ2)
	dW0 = np.dot(x.T, dZ1)

	w0 += dW0
	w1 += dW1

	if(j%1000 == 0):
		print("error value: "+str(np.mean(np.abs(dA2))))
