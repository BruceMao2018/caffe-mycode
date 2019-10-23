#coding=utf-8
#softmax分类器即对各个分类的得分进行exp(x)计算，然后求解各个分类的百分比
import numpy as np
def softmax(x):
	exp_x = np.exp(x)
	s = np.sum(exp_x, axis = 1, keepdims = True)
	return exp_x/s;

x = np.array([[2, 4],
		[1, 2],
		[3, 6]])

print (x)
print (softmax(x))
