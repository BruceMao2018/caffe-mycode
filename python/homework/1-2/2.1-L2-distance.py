#coding=utf-8
import numpy as np

#L2 distance又称为欧式距离，其公式为: (I1-I2)平方的平方累加，然后开根号??
def L2(yhat, y):
	loss = np.sum(np.power((yhat-y), 2))
	return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
