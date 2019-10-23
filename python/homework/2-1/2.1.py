#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils   #第一部分，初始化
import bruce
#import reg_utils    #第二部分，正则化
#import gc_utils     #第三部分，梯度校验
#%matplotlib inline #如果你使用的是Jupyter Notebook，请取消注释。
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""
实现功能: 	1, 生成一个点分布图，使用一个分类器将蓝点与红点进行分开
		2, 使用的是一个3层的神经网络 网络模型如下：
			LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
"""

#load数据，图形请参阅"点分布图.png"
train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=False)
layers_dims = (train_X.shape[0], 20, 3, 1)
#layers_dims = (train_X.shape[0], 10, 5, 1)

#X = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
#	          [0, 0, 1, 1, 0, 0, 1, 1],
#	          [0, 1, 0, 1, 0, 1, 0, 1]])

#Y = np.array([[0, 0, 0, 1, 0, 1, 1, 1]])
#train_X = X
#train_Y = Y
#layers_dims = (train_X.shape[0], 20, 3, 1)

print ("X.shape: " + str(train_X.shape))
print ("Y.shape: " + str(train_Y.shape))

parameters = bruce.L_layer_model_Reg(train_X, train_Y, layers_dims, learning_rate=0.01, num_iterations=30000, print_cost=True, initialization="he", isPlot=False, lambd=0.7, keep_prob=0.86)
print ("训练集:")
bruce.predict(train_X, train_Y, parameters)

print ("测试集:")
bruce.predict(test_X, test_Y, parameters)

#print (parameters)
