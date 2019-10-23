#coding=utf-8
import numpy as np

def sigmoid(x):
	return 1/(x+np.exp(-x)) #前向传播的时候，只要正常计算sigmoid的值，不需要求导

def sigmoid_backward(x):
	#print (x)
	return x * (1 - x)

def relu(x):
	return x * (x > 0)

def relu_backward(x):
	return (x > 0)

def initialize_parameters(n_x,n_h,n_y):
	"""
	输入:
		n_x - 单个样本特征数,即输入X的行数
		n_h - 隐藏层特征数(节点数)
		n_y - 输出值的特征数(节点数)
	输出：
        	parameters - 包含你的参数的python字典：
            		W1 - 权重矩阵,维度为（n_h，n_x）
            		b1 - 偏向量，维度为（n_h，1）
            		W2 - 权重矩阵，维度为（n_y，n_h）
            		b2 - 偏向量，维度为（n_y，1）
	"""
	W1 = 2 * np.random.random((n_h, n_x)) - 1
	b1 = np.zeros((n_h, 1))
	W2 = 2 * np.random.random((n_y, n_h)) - 1
	b2 = np.zeros((n_y, 1))

	parameters = {"W1": W1,
			"b1": b1,
			"W2": W2,
			"b2": b2}

	return parameters
