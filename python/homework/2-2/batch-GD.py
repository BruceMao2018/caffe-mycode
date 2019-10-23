#coding=utf-8

import numpy as np
import __init__
import bruce as bruce

"""
def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
	从（X，Y）中创建一个随机的mini-batch列表

	参数：
		X - 输入数据，维度为(输入节点数量，样本的数量)
		Y - 对应的是X的标签，【1 | 0】（蓝|红），维度为(1,样本的数量)
		mini_batch_size - 每个mini-batch的样本数量

	返回：
		mini-bacthes - 一个同步列表，维度为（mini_batch_X,mini_batch_Y）

	np.random.seed(seed) #指定随机种子
	m = X.shape[1]
	mini_batches = []

	#第一步：打乱顺序
	permutation = list(np.random.permutation(m)) #它会返回一个长度为m的随机数组，且里面的数是0到m-1
	shuffled_X = X[:,permutation]   #将每一列的数据按permutation的顺序来重新排列。
	shuffled_Y = Y[:,permutation].reshape((1,m))
	print ("shuffled_X: " + str(shuffled_X))
	print ("shuffled_Y: " + str(shuffled_Y))

	#第二步，分割
	#print("m: " + str(m))
	#print("size: " + str(mini_batch_size))
	num_complete_minibatches = m / mini_batch_size #把你的训练集分割成多少份,请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
	print ("num_complete_minibatches: " + str(num_complete_minibatches))
	for k in range(0,num_complete_minibatches):
		mini_batch_X = shuffled_X[:,k * mini_batch_size:(k+1)*mini_batch_size]
		mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k+1)*mini_batch_size]
		mini_batch = (mini_batch_X,mini_batch_Y)
		mini_batches.append(mini_batch)
		#print ("mini_batches: " + str(mini_batches))

	#如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
	#如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，我们要把它处理了
	if m % mini_batch_size != 0:
		#获取最后剩余的部分
		mini_batch_X = shuffled_X[:,mini_batch_size * num_complete_minibatches:]
		mini_batch_Y = shuffled_Y[:,mini_batch_size * num_complete_minibatches:]

		mini_batch = (mini_batch_X,mini_batch_Y)
		mini_batches.append(mini_batch)

	return mini_batches
"""

X = np.array([[1,2,3,4,5,6,7,8,9, 10],
				[1,2,3,4,5,6,7,8,9, 10],
				[1,2,3,4,5,6,7,8,9, 10]])
Y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
mini_batches = bruce.random_mini_batches(X,Y,mini_batch_size=3,seed=1)
print (mini_batches)
