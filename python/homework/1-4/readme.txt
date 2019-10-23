build a deep Nueral Networks
#重点理解sig1.py sig2.py dnn.py main.py

重点注意以下几个方面:
	1, cost函数的选择
	2, 计算dAL时的选择,不要简单的写成dAL = AL - Y
	3, 一定要加入惩罚项b,否则很容易出现梯度消失
	4, 反向传播中的激活函数参数，是传入A值，而非Z值
	5, 图片像素做归一化处理时，除以255.0，而不是255， 否则得到的值都是0

参数初始化:
W1 = 2 * np.random.random((4,3)) - 1 #初始化的权重参数建议分布在-1到1之间，np.random.random的
W2 = 2 * np.random.random((5,4)) - 1 #取值范围在0-1，故乘以2，再见1，即可获得-1到1之间的取值
W3 = 2 * np.random.random((2,5)) - 1 #取值范围在0-1，故乘以2，再见1，即可获得-1到1之间的取值
W4 = 2 * np.random.random((1,2)) - 1 #取值范围在0-1，故乘以2，再见1，即可获得-1到1之间的取值

"""
W1 =  np.random.randn(4, 3) / np.sqrt(3)
W2 =  np.random.randn(5, 4) / np.sqrt(4)
W3 =  np.random.randn(2, 5) / np.sqrt(5)
W4 =  np.random.randn(1, 2) / np.sqrt(2)
"""
b1 = np.zeros((4, 1))
b2 = np.zeros((5, 1))
b3 = np.zeros((2, 1))
b4 = np.zeros((1, 1))

前向传播：
	Z = np.dot(W, A_prev) + b
	A = sigmoid(Z)

dAL: #不要简单的把dAL写为AL - Y
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
反向传播:
	#AL = A - Y
	#dA = AL  (最后一层)

	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	dZ = dA * sigmoid'(A) -  !!!!!!!!!!!!!!!!注意是A值，不是Z值!!!!!!!!!!!!!!!!!
	dA_pre = np.dot(W.T, dZ)

	dW = np.dot(dZ, A_prev.T) / m
	db = np.sum(dZ, axis=1, keepdims=True) / m

	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)


cost:
	cost = -np.sum(np.multiply(np.log(A4),Y) + np.multiply(np.log(1 - A4), 1 - Y)) / A4.shape[1]
	cost = np.squeeze(cost)
	assert(cost.shape == ())



update参数时务必除以样品个数
	dW4 = (1.0/samples) * np.dot(dZ4, A3.T)
	db4 = (1.0/samples) * .....

db:  !!!!!!!!!!!!!!!!!!!!!!一定要设置惩罚项，否则会出现梯度消失的现象!!!!!!!!!!!!!!!!
	b1 = np.zeros((4, 1))
	db1 = np.sum(dZ1, axis=1, keepdims=True) / m
	assert (db.shape == b.shape)


机器学习中一个常见的预处理步骤是对数据集进行居中和标准化，这意味着可以减去每个示例中整个numpy数组的平均值，然后将每个示例除以整个numpy数组的标准偏差。但对于图片数据集，它更简单，更方便，几乎可以将数据集的每一行除以255（像素通道的最大值），因为在RGB中不存在比255大的数据，所以我们可以放心的除以255，让标准化的数据位于[0,1]之间，现在标准化我们的数据集
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.0
test_x = test_x_flatten/255.0

!!!!!!!!!!!!!!!!!!!!!!!!注意,一定要除以255.0，而不能是255这个整数，否则归一化的结果有很大变化，而且不准确!!!!!!!!!!!!!!!

关于正则化:
	正则化的原理是通过对参数W进行惩罚，限制W的参数，从而达到防止过拟合的情况
	正则化是否生效： 一个良好的正则化方式会使得训练集精度下降，测试集精度上升
	
	正则化的影响范围:
		1, 前向传播时，成本计算时， 需要加入正则惩罚项
		2, 反向传播时，需要考虑正则化对dW的影响, 正则化不影响其它参数
			dW = 1.0 * np.dot(dZ, A_prev.T) / m + ((lambd * W) / m)

关于dropout:
	dropout主要有两个方面的影响： 1，前向传播计算A值时，需要考虑dropout，并将A值除以keep_prob
								  2, 反向传播时影响dA的值
									即：dropout只影响前向中的A值及反向中的dA值
									但是注意： 最后一层不使用dropout
