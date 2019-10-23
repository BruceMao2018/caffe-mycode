# -*- coding: utf-8 -*-
"""
Created on 2019-09-12

@author: Bruce
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy 
from PIL import Image
from scipy import ndimage
from load_data import load_dataset

# Loading the data (cat/non-cat)
#使用load_data后读取的数据格式为: 样本数 X 像素高度 X 像素宽度 X 通道数
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

"""
print ("训练集大小: " + str(train_set_x_orig.shape))
print ("训练集标签大小: " + str(train_set_y.shape))
print ("测试集大小: " + str(test_set_x_orig.shape))
print ("测试集标签大小: " + str(test_set_y.shape))
print ("分类：" + str(classes))
"""
index = 25
#plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

"""
Exercise: Find the values for: 
- m_train (number of training examples) 
- m_test (number of test examples) 
- num_px (= height = width of a training image) 
Remember that train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3). For instance, you can access m_train by writing train_set_x_orig.shape[0].
"""
n1 = train_set_x_orig.shape[0]
n2 = test_set_x_orig.shape[0]
print ("训练集个数: " + str(n1) + " 测试集个数: " + str(n2))
print ("训练集图像尺寸: " + str(train_set_x_orig.shape[1]) + " X " + str(train_set_x_orig.shape[2]) + " X " + str(train_set_x_orig.shape[3]))

"""
Exercise: Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into single vectors of shape (num_px [Math Processing Error]∗ num_px [Math Processing Error]∗ 3, 1).
将训练集中的所有图像进行展开，转换成(nx,m)的样式，以便进行前向反向传播
"""

train_set_x_flatten = train_set_x_orig.reshape(n1, -1).T #将原始数据按n1行进行展开，因为样本个数作为矩阵列的值，故将展开后的矩阵进行转置
print ("预备训练数据: " + str(train_set_x_flatten.shape))

test_set_x_flatten = test_set_x_orig.reshape(n2, -1).T #将原始数据按n1行进行展开，因为样本个数作为矩阵列的值，故将展开后的矩阵进行转置
print ("预备训练数据: " + str(test_set_x_flatten.shape))

"""
One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).

Let’s standardize our dataset.
"""

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

"""
What you need to remember:

Common steps for pre-processing a new dataset are: 
- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, …) 
- Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1) 
- “Standardize” the data
"""
