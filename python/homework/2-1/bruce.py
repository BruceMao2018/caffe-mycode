# -*- coding: utf-8 -*-
import numpy as np
import gradient_check

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(A):
    return A * (1 - A)

def relu(Z):
    return Z * (Z > 0)

def relu_backward(A):
    return A > 0

def initialize_parameters(layers_dims, type="random"):
	"""
	功能： 为多层神经网络的权值进行初始化
		初始化为0：在输入参数中全部初始化为0，参数名为initialization = “zeros”，核心代码： 
		parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
		初始化为随机数：把输入参数设置为随机值，权重初始化为大的随机值。参数名为initialization = “random”，核心代码： 
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
		抑梯度异常初始化：参见梯度消失和梯度爆炸的那一个视频，参数名为initialization = “he”，核心代码： 
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])
	输入: layers_dims, 神经网络各层节点数(特征数) e.g (12888, 20, 7, 5, 1)
		type, 初始化的方法，取值范围"zero", "random", "he"
	输出: parameters - e.g {"W1": W1, "b1":b1, "W2":W2, "b2":b2, ...}
	"""
	np.random.seed(3)
	parameters = {}
	L = len(layers_dims) - 1
	for i in range(L):
		if(type == "zero"):
			parameters["W"+str(i+1)] = np.zeros((layers_dims[i+1], layers_dims[i]))
			parameters["b"+str(i+1)] = np.zeros((layers_dims[i+1], 1))
		elif(type == "random"):
			np.random.seed(1)
			parameters["W"+str(i+1)] = np.random.random((layers_dims[i+1], layers_dims[i])) * 2 - 1 #以0为中心，取值范围在[-1,1]
			#parameters["W"+str(i+1)] = np.random.randn(layers_dims[i+1], layers_dims[i]) * 10 #缩放10倍，测试用，实际应用中不会放大
			parameters["b"+str(i+1)] = np.zeros((layers_dims[i+1], 1))
		elif(type == "he"):
			#正态分布，0为中心，方差为1，即n(0, 1)
			parameters["W"+str(i+1)] = np.random.randn(layers_dims[i+1], layers_dims[i]) * np.sqrt(1.0/layers_dims[i]) #开根号中的值是当前层的节点数, 并且应该使用乘，使用除的话容易得到的是0值, 另外注意使用浮点数，不要使用整数
			#parameters["W"+str(i+1)] = np.random.randn(layers_dims[i+1], layers_dims[i]) / np.sqrt(layers_dims[i])  #开根号中的值是当前层的节点数, 并且应该使用乘，使用除的话容易得到的是0值
			parameters["b"+str(i+1)] = np.zeros((layers_dims[i+1], 1))
		else:
			print("init parameters error")
			return

	return parameters

def linear_forward(A_prev,W,b):
	"""
	实现前向传播的线性部分。
	参数：
		A_prev - 来自上一层（或输入数据）的激活，维度为(上一层的节点数量，示例的数量）
		W - 本层的权重矩阵，numpy数组，维度为（当前图层的节点数量，前一图层的节点数量）
		b - 本层的偏向量，numpy向量，维度为（当前图层节点数量，1）

	返回 - (cache):
		Z - 本层的Z值
	"""
	Z = np.dot(W, A_prev) + b
	assert(Z.shape == (W.shape[0], A_prev.shape[1]))

	#返回本层的Z值
	return Z

def linear_activation_forward(A_prev,W,b,activation, keep_prob):
	"""
	实现LINEAR-> ACTIVATION 这一层的前向传播
	参数：
		A_prev - 来自上一层（或输入层）的激活，维度为(上一层的节点数量，示例数）
		W - 权重矩阵，numpy数组，维度为（当前层的节点数量，前一层的大小）
		b - 偏向量，numpy阵列，维度为（当前层的节点数量，1）
		activation - 选择在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
		keep_prob - 表示dropout，值为1时表示不使用dropout(或者是最后一层)

	返回：
		A - 激活函数的输出，也称为激活后的值
	"""
	np.random.seed(1)
	Z = linear_forward(A_prev, W, b)
	if (activation == "sigmoid"):
		A = sigmoid(Z)
	elif activation == "relu":
		A = relu(Z)

	#以下方式可以cover keep_prob==1,不做dropout的情形
	D = np.random.random((A.shape[0], A.shape[1])) #不要使用random.randn正态分布，因为randn会产生大于1的值.使用正态分布可能产生一个问题：1，最后一层不能使用keep_prob==1来表示不需要dropout，因为正态分布的值可能会大于1，这样导致达不到不做dropout的效果
	#D = np.random.rand(A.shape[0], A.shape[1]) #不要使用random.randn正态分布，因为randn会产生大于1的值.使用正态分布可能产生一个问题：1，最后一层不能使用keep_prob==1来表示不需要dropout，因为正态分布的值可能会大于1，这样导致达不到不做dropout的效果
	D = D < keep_prob
	A = A * D
	A = A / keep_prob
	
	assert(A.shape == (W.shape[0],A_prev.shape[1]))

	#返回本层的A值， 及D值
	return A, D

def L_model_forward(X,parameters, keep_prob):
	"""
	实现[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION
	参数：
		X - 数据，numpy数组，维度为（输入节点数量，示例数）
		parameters - initialize_parameters_deep（）的输出

	返回：
		AL - 最后的激活值
		caches - 包含以下内容的缓存列表：(A, D, W, b)
	"""
	caches = []
	A = X
	L = len(parameters)//2 #层数应该是paras的长度除以2,因为里面包含了w，b两个参数
	for l in range(1,L): #进行前面n-1层的前向传播
		A_prev = A 
		A, D = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu", keep_prob)
		cache = (A, D, parameters["W" + str(l)], parameters["b" + str(l)])
		#print ("cache["+str(l-1)+"]: " + str(cache))
		caches.append(cache)

	#对最后一层进行前向传播
	#!!!!!!!!!注意-最后一层不使用dropout!!!!!!!!!!!!
	AL, D = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid", 1)
	cache = (AL, D, parameters["W" + str(L)], parameters["b" + str(L)])
	caches.append(cache)

	#print ("AL.shape: " + str(AL.shape))
	#print ("X.shape1: " + str(X.shape[1]))
	assert(AL.shape == (1, X.shape[1]))

	return AL,caches

def compute_cost(AL,Y):
	"""
	实施等式（4）定义的成本函数。

	参数：
		AL - 与标签预测相对应的概率向量，维度为（1，示例数量）
		Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

	返回：
		cost - 交叉熵成本
	"""
	m = Y.shape[1]
	cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) * 1.0 / m
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	return cost

def compute_cost_with_Reg(AL, Y, lambd, caches):
	"""
	加入L2正则惩罚项
	参数：
		AL - 与标签预测相对应的概率向量，维度为（1，示例数量）
		Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）
		caches - 包含所有层的(A, D, W, b)缓存列表

	返回：
		cost - 交叉熵成本
	"""
	cost = compute_cost(AL, Y)
	
	m = Y.shape[1] #样本个数
	L = len(caches) #层数
	lost = 0;
	for i in range(L):
		#lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2))  + np.sum(np.square(W3))) / (2 * m)
		lost += np.sum(np.square(caches[i][2]))

	L2_cost = cost + lambd * lost / (2 * m)

	

	return L2_cost

def linear_backward(dZ,cache, A_prev, lambd):
	"""
	根据本层的dZ及W，求解前一层的dA值,dA_prev  - dA_prev = dZ * W.T
	根据本层的dZ及上一层的A值,A_prev，求解本层的dw， dw = dZ * A_prev.T

	输入：
		dZ - 本层的dZ
		cache - 来自当前层前向传播的值的元组(A, D, W, b)
		A_prev - 上一层的A值

	返回：
		dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
		dW - 相对于W（当前层l）的成本梯度，与W的维度相同
		db - 相对于b（当前层l）的成本梯度，与b维度相同
	"""
	A, D, W, b = cache
	m = A_prev.shape[1]
	dW = 1.0 * np.dot(dZ, A_prev.T) / m + ((lambd * W) / m)
	db = 1.0 * np.sum(dZ, axis=1, keepdims=True) / m
	dA_prev = np.dot(W.T, dZ)

	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	#assert (db.shape == b.shape)

	return dA_prev, dW, db

def linear_activation_backward(dA, cache, A_prev, activation="relu", lambd=0, keep_prob=1, lastLayer=False):
	"""
	功能:
		1, 根据本层的dA，D值, keep_prob,求解dropout后的dA
		2, 根据1中的dA及本层的A值，求解本层的dZ， dZ = dA * sigmoid'(A)
		3, 根据本层的dZ及上一层的A值(A_prev)求解dW
		4, 根据dZ，求解db
		5, 根据本层的dZ及W，求解上一层的dA(dA_prev)

	参数：
		dA - 当前层的dA值 - 未经过dropout
		cache - 当前层的(A,D,W,b)
		A_prev - 上一层的A值,如果当前层是第一层的话，此为输入X
		activation - 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
		lambd - 正则化参数
		keep_prob - dropout参数
		lastLayer - True:最后一层， others - No
	返回：
		dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
		dW - 本层的dW，如果使用了正则化，则会返回计算了正则化后的值
		db - 相对于b（当前层l）的成本梯度值，与b的维度相同
	"""
	A, D, W, b = cache
	#print("in cache: A.shape: " + str(A.shape) + " D.shape: " + str(D.shape) + " W.shape: " + str(W.shape) + " b.shape: " + str(b.shape))
	if( lastLayer == False and keep_prob < 1): #除最后一层以外，都需要做dropout
		dA = dA * D
		dA = dA / keep_prob

	if activation == "relu":
		dZ = dA * relu_backward(A)
	elif activation == "sigmoid":
		dZ = dA * sigmoid_backward(A)

	assert (dZ.shape == dA.shape)
	dA_prev, dW, db = linear_backward(dZ, cache, A_prev, lambd)

	return dA_prev,dW,db

#just copy from here
def L_model_backward(AL, X, Y, caches, lambd, keep_prob):
	"""
	多层网络的反向传播

	参数：
		AL - 概率向量，正向传播的输出(L_model_forward())
		Y - 标签向量(例如：如果不是猫，则为0，如果是猫则为1)，维度为(1，数量)
		caches - 包含每一层的A,D,W,b, e.g ((A1, D1, W1, b1), (A2, D2, W2, b2), ....(Am, Dm, Wm, bm))
		lambd - 0-未使用L2正则化, others-使用L2正则化-需要考虑lambd对dW的影响
		keep_prob - dropout参数
	返回：
		grads - 具有梯度值的字典
			grads [“dA”+ str（l）] = ...
			grads [“dW”+ str（l）] = ...
			grads [“db”+ str（l）] = ...
	"""
	grads = {}
	L = len(caches)
	#print ("we total have " + str(L) + " layers here !")
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	#dAL = AL - Y

	current_cache = caches[L-1]
	A_prev = caches[L-2][0]
	#最后一层不使用dropout, 下列函数返回上一层的dA值，及本层的dW，db值
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, A_prev, "sigmoid", lambd, keep_prob, lastLayer=True)

	for l in reversed(range(L-1)): # 2, 1, 0
		if (l == 0):
			A_prev = X
		else:
			A_prev = caches[l-1][0]

		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, A_prev, "relu", lambd, keep_prob, lastLayer=False)
		#current_cache = caches[l]
		#pre_cache = caches(l-1)
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads

def update_parameters(parameters, grads, learning_rate):
	"""
	使用梯度下降更新参数

	参数：
		parameters - 包含你的参数的字典
		grads - 包含梯度值的字典，是L_model_backward的输出

	返回：
		parameters - 包含更新参数的字典
			参数[“W”+ str（l）] = ...
			参数[“b”+ str（l）] = ...
	"""
	L = len(parameters) // 2 #整除
	for l in range(L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

	return parameters

def L_layer_model_Reg(X, Y, layers_dims, learning_rate=0.01, num_iterations=2000, print_cost=False, initialization="he", isPlot=False, lambd=0, keep_prob=1):
	"""
	实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID。
	使用L2正则惩罚项

	参数：
		X - 输入的数据，维度为(n_x，例子数)
		Y - 标签，向量，0为非猫，1为猫，维度为(1,数量)
		layers_dims - 层数的向量，维度为(n_y,n_h,···,n_h,n_y)
		learning_rate - 学习率
		num_iterations - 迭代的次数
		print_cost - 是否打印成本值，每100次打印一次
		isPlot - 是否绘制出误差值的图谱
		lambd - 正则化的超参数，实数
		keep_prob - 随机删除节点的概率

	返回：
		parameters - 模型学习的参数。 然后他们可以用来预测。
	"""
	np.random.seed(1)
	costs = []

	parameters = initialize_parameters(layers_dims, initialization)
	#print ("初始化参数: " + str(parameters))

	for i in range(0,num_iterations):
		#keep_prob - 1,表示未使用dropout
		#Reg = 'L2', 表示使用L2正则惩罚项，为空时表示未使用正则惩罚项
		AL , caches = L_model_forward(X,parameters, keep_prob)
		#print (AL)
		if( lambd == 0): #未使用L2正则惩罚项
			cost = compute_cost(AL,Y)
		else:
			cost = compute_cost_with_Reg(AL,Y, lambd, caches) #加入L2正则惩罚项
		#caches - 包含以下内容的缓存列表：(A, D, W, b)
		grads = L_model_backward(AL, X, Y, caches, lambd, keep_prob)
		#if (i % 1000 == 0):
		if (i % 10 == 0):
			#grandient check
			gradient_check.gradient_check(parameters,grads,X,Y,epsilon=1e-7)

		parameters = update_parameters(parameters,grads,learning_rate)

		#打印成本值，如果print_cost=False则忽略
		if i % 1000 == 0:
			#记录成本
			costs.append(cost)
			#是否打印成本值
			if print_cost:
				print("第"+str(i) +"次迭代，成本值为：" +str(np.squeeze(cost)))
				#print ("第"+str(i+1)+"次进行迭代: AL: " + str(AL))

	#迭代完成，根据条件绘制图
	if isPlot:
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
	return parameters

def predict(X, y, parameters):
	"""
	该函数用于预测L层神经网络的结果，当然也包含两层

	参数：
		X - 测试集
		y - 标签
		parameters - 训练模型的参数

	返回：
		p - 给定数据集X的预测
	"""

	m = X.shape[1]
	n = len(parameters) // 2 # 神经网络的层数
	p = np.zeros((1,m), dtype = np.int)

	#根据参数前向传播
	probas, caches = L_model_forward(X, parameters, 1)

	#print ("预测值: " + str(probas))

	for i in range(0, probas.shape[1]):
		if probas[0,i] > 0.50:
			p[0,i] = 1
		else:
			p[0,i] = 0

	print ("p: " + str(p))
	print ("y: " + str(y))
	print ("准确度为: "  + str(float(np.sum((p == y))/m)))
	print ("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))

	return p
