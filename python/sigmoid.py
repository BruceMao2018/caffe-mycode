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
#		___		___	___	___
#		| |	 _ 	| |	|_|	|_|
#	x1	|_|	| |	|_|S	|_|	|_|
#	 	| |	|_|S	| |i	|_|	|_|
#	x2	|_|	| |i	|_|g	|_|	|_|
#	 	| |	|_|g	| |m	|_|	|_|
#	x3	|_|	| |m	|_|o	|_|	|_|
#	 	| |	|_|o	| |i	|_|	|_|
#	x4	|_|	| |i	|_|d	|_|	|_|
#	 	| |	|_|d	| |	|_|	|_|
#	x5	|_|	   	|_|	|_|	|_|
#
#
#	x	|_|	|_|	|_|	|_|	|_|
#
import numpy as np
def sigmoid(x, deriv = False):
	if( deriv == True):   #true means Back-propagation, need to get deriv, false means Forward-propagation, no need deriv
		return x*(1-x) #注意：此中的x是指前向传播经过sigmoid计算后的值，并非前向传播中的原始输入值x
	return 1/(x+np.exp(-x)) #前向传播的时候，只要正常计算sigmoid的值，不需要求导

x = np.array([[0,0,1],
				[0,1,1],
				[1,0,1],
				[1,1,1],
				[0,0,1]])
print (x.shape) #(5,3)-意味着一共有5个样本，每个样本有3个特征

y = np.array([[0],
			 [0],
			 [0],
			 [1],
			 [0]])
print (y.shape) #(5,1), y表示上述5个样本的正确输出值，即label标签

np.random.seed(1) #指定随机种子，这样的话每次随机的时候取值就是一样的，便于比较

w0 = 2 * np.random.random((3,4)) - 1 #初始化的权重参数建议分布在-1到1之间，np.random.random的
w1 = 2 * np.random.random((4,1)) - 1 #取值范围在0-1，故乘以2，再见1，即可获得-1到1之间的取值
print (w0)

for j in xrange(5000):
#前向传播开始
	l0 = x #输入层
	l1 = np.dot(l0, w0)
	#每一层加入激活函数
	l1 = sigmoid(l1)

	
	l2 = np.dot(l1, w1)
	#每一层加入激活函数
	l2 = sigmoid(l2)
#前向传播结束

#计算l2与实际输出y的误差,即lost值
	l2_error = y - l2
	#print (l2_error.shape)
	if( j%1000) == 0:
		print("error value: "+str(np.mean(np.abs(l2_error))))

#反向传播开始
	#反向传播的第一步是激活函数sigmoid
	#根据lost值l2_error来更新l2_delta-即更新量（差值),根据差值再来更新l2
	l2_delta = l2_error * sigmoid(l2, deriv=True) #此处使用对应相乘，非矩阵相乘
	#l2 = l2 - l2_delta
	#print (l2_delta.shape)
	#print (sigmoid(l2,deriv=True).shape)

	#l2层更新结束，继续反向传播，到达l1层,(l1层同样具有激活函数)
	l1_error = np.dot(l2_delta, w1.T) #前一层的loss值等于当前层的偏移量与前一层当前层的权重的转自-即矩阵求导
	#l1_error = l2_error * sigmoid(l2, deriv=True) * w1.T
	l1_delta = l1_error * sigmoid(l1, deriv=True)

	#更新权重参数,考虑到l2 = l1 * w1, 当前的l2与l1已经更新，故可以依此公式来更新w1
	#考虑到w参数的更新是差值，故l2_delta = l1 * w1_delta , 推导出w1_delta = l1.T * l2_delta,即斜切率乘以差值
	w1_delta = np.dot(l1.T, l2_delta)
	w0_delta = np.dot(l0.T,l1_delta)

	w1 += w1_delta #w1更新时是使用+=还是-=？由l2_error = y -l2 或者 l2_error = l2 - y 来决定的。
			#试想一下，使用l2_error = y - l2, 如果l2_error非常大，那么意味着l2的值非常小，则需要加大参数w，以便l2获取更大的值,从而达到l2_error趋向于0的结果
			#同理，如果使用 l2 - y, 则在更新w参数时需要使用-=
	w0 += w0_delta
#反向传播结束
