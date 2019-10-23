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
#	         X    w1 L1  w2 L2
#		___		___	___	___
#		| |	 _ 	| |	|_|	|_|
#	x1	|_|	| |	| |S	|_|	|_|
#	 	| |	|_|S	| |i	|_|	|_|
#	x2	|_|	| |i	| |g	|_|	|_|
#	 	| |	|_|g	| |m	|_|	|_|
#	x3	|_|	| |m	| |o	|_|	|_|
#	 	| |	|_|o	| |i	|_|	|_|
#	x4	|_|	| |i	| |d	|_|	|_|
#	 	| |	|_|d	| |	|_|	|_|
#	x5	|_|	   	|_|	|_|	|_|
#		    w1=(4, 3)   w2=(1, 4)
#		X=(3, 5) L1=(4, 5) L2=(1, 5)
#
import numpy as np
def sigmoid(x, deriv = False):
	if( deriv == True):   #true means Back-propagation, need to get deriv, false means Forward-propagation, no need deriv
		return x*(1-x) #注意：此中的x是指前向传播经过sigmoid计算后的值，并非前向传播中的原始输入值x
	return 1/(x+np.exp(-x)) #前向传播的时候，只要正常计算sigmoid的值，不需要求导

x = np.array([[0, 0, 1, 1, 0],
		[0, 1, 0, 1, 0],
		[1, 1, 1, 1, 1]])
print (x.shape) #(3,5)-意味着一共有5个样本，每个样本有3个特征

y = np.array([[0, 0, 0, 1, 0]])
print (y.shape) #(1,5), y表示上述5个样本的正确输出值，即label标签

np.random.seed(1) #指定随机种子，这样的话每次随机的时候取值就是一样的，便于比较

w1 = 2 * np.random.random((4,3)) - 1 #初始化的权重参数建议分布在-1到1之间，np.random.random的
w2 = 2 * np.random.random((1,4)) - 1 #取值范围在0-1，故乘以2，再见1，即可获得-1到1之间的取值
print (w1)

for j in xrange(1000):
#前向传播开始
	l0 = x #输入层
	Z1 = np.dot(w1, x)
	#每一层加入激活函数
	A1 = sigmoid(Z1)

	
	Z2 = np.dot(w2, A1)
	#每一层加入激活函数
	A2 = sigmoid(Z2)
#前向传播结束

#计算l2与实际输出y的误差,即lost值
	dA2 = y - A2
	#print (l2_error.shape)
	if( j%1000) == 0:
		print("error value: "+str(np.mean(np.abs(dA2))))

#反向传播开始
	#反向传播的第一步是激活函数sigmoid
	dZ2 = dA2 * sigmoid(A2, deriv=True) #此处使用对应相乘，非矩阵相乘

	#l2层更新结束，继续反向传播，到达l1层,(l1层同样具有激活函数)
	dA1 = np.dot(w2.T, dZ2) #前一层的loss值等于当前层的偏移量与前一层当前层的权重的转自-即矩阵求导
	dZ1 = dA1 * sigmoid(A1, deriv=True)

	dw2 = np.dot(dZ2, A1.T)
	dw1 = np.dot(dZ1, x.T)

	w2 += dw2
	w1 += dw1

print (A2)
print (y)

#反向传播结束
