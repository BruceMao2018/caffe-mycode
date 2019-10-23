#coding=utf-8
#	试验目的:使用单步方式实现前向，反向传播，与t1.py and dnn.py进行比较
#	尝试加入l2正则惩罚项
#	Bruce.Mao
#	Version 01	2019.09.19
#	5个输入样本，每个样本3个特征，及输入层X为(3,5)的矩阵
#
#
#		---> W1(4, 3)		--->W2(5, 4)		--->W3((2, 5)		--->W4(1, 2)
#	X=(3, 5)  	L1=(4, 5)-relu		L2=(5, 5)-relu		L3=(2, 5)-sigmoid	L4=(1, 5)-sigmoid   - Y
#
#	   	L1 	L2 	L3	L4
#	  ->W1	  ->W2	 _->W3	   ->W4	   
#		   	| |   	   	   	
#	x1	 _ 	|_|	 _ S	 _ 
#	 	| |S	| |S	| |i	| |S
#	x2	|_|i	|_|i	| |g	| |i
#	 	| |g	| |g	| |m	| |g
#	x3	|_|m	|_|m	|_|o	| |m
#	 	| |o	| |o	| |i	|_|o
#	x4	|_|i	|_|i	| |d	   i
#	 	| |d	| |d	| |	   d
#	x5	|_|	|_|	|_|	   
#
#
#
import numpy as np
np.random.seed(1) #指定随机种子，这样的话每次随机的时候取值就是一样的，便于比较

samples = 5 #总计5个样本
layer_num = 4 #4层网络
lambd = 0


def sigmoid(A, deriv = False):
	if( deriv == True):   #true means Back-propagation, need to get deriv, false means Forward-propagation, no need deriv
		return A*(1-A) #注意：此中的x是指前向传播经过sigmoid计算后的值，并非前向传播中的原始输入值x
	return 1/(1+np.exp(-A)) #前向传播的时候，只要正常计算sigmoid的值，不需要求导

def relu(A, deriv = False):
	if( deriv == True): #反向
		return ( A > 0 )
	return A * ( A > 0) #前向

X = np.array([[0, 1, 0, 0, 1],
	      [0, 1, 0, 0, 1],
	      [0, 1, 0, 0, 1]])
Y = np.array([[0, 1, 0, 0, 1]])

W1 =  np.random.randn(4, 3) / np.sqrt(3) * 1 #在做W初始化的时候，randn返回的是一个以0为中心，1为方差的正态分布值，记为: n(0, 1)
W2 =  np.random.randn(5, 4) / np.sqrt(4) * 1 #正态分布值为什么还要除以每层的节点个数???
W3 =  np.random.randn(2, 5) / np.sqrt(5) * 1 #对于包含Relu的层，建议乘以2，其他乘以1
W4 =  np.random.randn(1, 2) / np.sqrt(2) * 1

print (W1)
print (W2)
print (W3)
print (W4)

"""
W1 =  2 * np.random.random((4, 3)) - 1 #参数分布在-1到1之间的随机数，但非正态分布
W2 =  2 * np.random.random((5, 4)) - 1
W3 =  2 * np.random.random((2, 5)) - 1
W4 =  2 * np.random.random((1, 2)) - 1
"""
b1 = np.zeros((4, 1))
b2 = np.zeros((5, 1))
b3 = np.zeros((2, 1))
b4 = np.zeros((1, 1))


num = 5000
for j in xrange(num):
#前向传播开始
	l0 = X #输入层
	Z1 = np.dot(W1, X) + b1
	A1 = relu(Z1)
	
	Z2 = np.dot(W2, A1) + b2
	A2 = relu(Z2)

	Z3 = np.dot(W3, A2) + b3
	A3 = relu(Z3)
	
	Z4 = np.dot(W4, A3) + b4
	A4 = sigmoid(Z4)

	#if( j%1000 == 0):
		#print ("完成"+str(j+1)+"次迭代后的AL: " + str(A4))
#前向传播结束


#反向传播开始
#计算l2与实际输出y的误差,即lost值
	#dA4 = A4 - Y
	dA4 = - (np.divide(Y, A4) - np.divide(1 - Y, 1 - A4))
	if( j%(1000) ) == 0:
		cost = -np.sum(np.multiply(np.log(A4),Y) + np.multiply(np.log(1 - A4), 1 - Y)) / A4.shape[1]
		l2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)) + np.sum(np.square(W4))) / A4.shape[1] / layer_num
		cost += l2_regularization_cost
		cost = np.squeeze(cost)
		assert(cost.shape == ())
		print("error value: "+ str(cost))

	#反向传播的第一步是激活函数sigmoid
	dZ4 = dA4 * sigmoid(A4, deriv=True)
	dA3 = np.dot(W4.T, dZ4)
	assert( dZ4.shape == Z4.shape )
	assert( dA3.shape == A3.shape )

	dZ3 = dA3 * relu(A3, deriv=True)
	dA2 = np.dot(W3.T, dZ3)
	assert( dZ3.shape == Z3.shape )
	assert( dA2.shape == A2.shape )

	dZ2 = dA2 * relu(A2, deriv=True)
	dA1 = np.dot(W2.T, dZ2)
	assert( dZ2.shape == Z2.shape )
	assert( dA1.shape == A1.shape )

	dZ1 = dA1 * relu(A1, deriv=True)
	assert( dZ1.shape == Z1.shape )

#反向传播结束
	
#update参数	
	samples = A4.shape[1]
	dW4 = (1.0/samples) * np.dot(dZ4, A3.T) + lambd / samples * W4
	dW3 = (1.0/samples) * np.dot(dZ3, A2.T) + lambd / samples * W3
	dW2 = (1.0/samples) * np.dot(dZ2, A1.T) + lambd / samples * W2
	dW1 = (1.0/samples) * np.dot(dZ1, X.T) + lambd / samples * W1
	
	db1 = np.sum(dZ1, axis=1, keepdims=True) / samples
	db2 = np.sum(dZ2, axis=1, keepdims=True) / samples
	db3 = np.sum(dZ3, axis=1, keepdims=True) / samples
	db4 = np.sum(dZ4, axis=1, keepdims=True) / samples

	assert(dW4.shape == W4.shape)
	assert(dW3.shape == W3.shape)
	assert(dW2.shape == W2.shape)
	assert(dW1.shape == W1.shape)

	W4 = W4 - 0.01 * dW4
	W3 = W3 - 0.01 * dW3
	W2 = W2 - 0.01 * dW2
	W1 = W1 - 0.01 * dW1

	b1 = b1 - 0.01 * db1
	b2 = b2 - 0.01 * db2
	b3 = b3 - 0.01 * db3
	b4 = b4 - 0.01 * db4

	assert(dW4.shape == W4.shape)
	assert(dW3.shape == W3.shape)
	assert(dW2.shape == W2.shape)
	assert(dW1.shape == W1.shape)

print (A4)
