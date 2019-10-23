#coding=utf-8
#L1 distance计算公式为: d(I1, I2) = sum(I1 - I2)

import numpy as np

def L1(yhat, y): #yhat为输出，y为真实标签
	loss = np.sum(np.abs(yhat - y))
	return loss

x = np.array([[1, 2, 3]])
y = np.array([[2, 4, 6]])

print (L1(x, y))
