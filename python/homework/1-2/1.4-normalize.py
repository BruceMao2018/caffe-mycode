#coding=utf-8
#1.4 - Normalizing rows
#Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to x∥x∥x‖x‖ (dividing each row vector of x by its norm).
#Exercise: Implement normalizeRows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).
#实现对矩阵按行求2范数
#2范数的求解公式为:单个值的平方求和再开根号,如果是按行求范数，则同一行所有值的平方求和再开根号

import numpy as np

#x_norm=np.linalg.norm(x, ord=None, axis=None, keepdims=False)
def linaRow(x):
	s = np.linalg.norm(x, ord = 2, axis = 1, keepdims=True)
	return x/s

def linaCol(x):
	s = np.linalg.norm(x, ord = 2, axis = 0, keepdims = True)
	return x/s

x = np.array([[0, 3, 4],
		[1, 6, 4]])

print ("x的矩阵\n" + str(x))
print ("按行使用2范数归一化:\n" + str(linaRow(x)))
print ("按列使用2范数归一化:\n" + str(linaCol(x)))
