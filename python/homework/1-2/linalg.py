#coding=utf-8
#x_norm=np.linalg.norm(x, ord=None, axis=None, keepdims=False)

import numpy as np
x = np.array([[0, 3, 4],
		[2, 6, 4]])

print (x.shape)

linalg = np.linalg.norm(x)
print "默认参数(矩阵2范数，不保留矩阵二维特性): " + str(np.linalg.norm(x))
print "默认参数(矩阵2范数，保留矩阵二维特性): " + str(np.linalg.norm(x, ord=2, axis=None, keepdims=True))
print "矩阵每个行向量求向量的2范数：" + str(np.linalg.norm(x,ord = 2, axis=1, keepdims=True))
print "矩阵每个列向量求向量的2范数：" + str(np.linalg.norm(x,ord = 2, axis=0, keepdims=True))

print "默认参数(矩阵1范数，不保留矩阵二维特性): " + str(np.linalg.norm(x, ord=1))
print "默认参数(矩阵1范数，保留矩阵二维特性): " + str(np.linalg.norm(x, ord=1, axis=None, keepdims=True))
print "矩阵每个行向量求向量的1范数：" + str(np.linalg.norm(x,ord = 1, axis=1, keepdims=True))
print "矩阵每个列向量求向量的1范数：" + str(np.linalg.norm(x,ord = 1, axis=0, keepdims=True))
