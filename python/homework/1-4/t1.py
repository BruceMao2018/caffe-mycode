#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from dnn_utils import *

"""
本程序主要实现5个样本，每个样本3个特征
总计4层，前两层使用relu，后两层使用sigmoid进行二分类

	---> W1(4, 3)		--->W2(5, 4)		--->W3((2, 5)		--->W4(1, 2)
X=(3, 5)  	L1=(4, 5)-relu		L2=(5, 5)-relu		L3=(2, 5)-sigmoid	L4=(1, 5)-sigmoid   - Y

"""

np.random.seed(1)
m = 5
layers = 4

def relu(x):
	return x * (x > 0)

def compute_cost(AL,Y):
	"""
	参数：
		AL - 与标签预测相对应的概率向量，维度为（1，示例数量）
		Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）

	返回：
		cost - 交叉熵成本
	"""
	assert(AL.shape == Y.shape)
	#cost = (1.0/m) * np.sqrt(np.sum(np.power((AL - Y), 2)))
	cost = -np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1 - AL), 1 - Y)) / m
	assert(cost.shape == ())

	return cost


def one_layer_forward(W, b, last_A, Act):
	"""
	输入:
		last_A - 前一层激活函数后的值
		W : 本层的权重参数
		Act:本层的激活函数
			
	输出:
		Z - 线性值
		A - 线性激活值
	"""
	Z = np.dot(W, last_A) + b
	if( Act == "sigmoid" ):
		A = sigmoid(Z)
	elif( Act == "relu" ):
		A = relu(Z)
	else:
		print ("未定义的激活函数")
		exit(-1)

	return Z, A

def one_layer_backward(Z, W, b, dA, last_A, Act):
	"""
	输入:
		Z   - 本层前向传播的Z值,用于求dZ
		W,b - 本层的W,b值
		dA - 本层的delta
		last_A - 前层的A值
		Act - 本层使用的激活函数
	输出:
		dZ - 本层的激活函数后的delta
		dw - 本层W的更新权值
		db - 本层b更新权值
		last_dA - 前一本层的激活函数前delta
	"""

	if( Act == "sigmoid" ):
		dZ = dA * sigmoid_backward(sigmoid(Z)) #注意，在反向传播中求dZ时,sigmoid_backward中传入的参数是前向传播的A值，而并不是dA或者Z
	elif( Act == "relu" ):
		dZ = dA * relu_backward(relu(Z)) #注意，在反向传播中求dZ时,sigmoid_backward中传入的参数是前向传播的A值，而并不是dA或者Z
	else:
		print("Act error")

	last_dA = np.dot(W.T, dZ)
	assert(last_dA.shape == last_A.shape)

	dw = (1.0/5) * np.dot(dZ, last_A.T) #hardcode - m = 5 - 样本数
	db = np.sum(dZ, axis=1, keepdims=True) / m
	#db = (1.0/5) * np.sum(dA) #hardcode - m = 5
	#db = 0
	assert(W.shape == dw.shape)
	#assert(db.dtype == float)

	return dZ, dw, db, last_dA

def all_layers_backward(paras, layers, cache, X, Y):
	"""

	输入:	paras = initialize_parameters()的返回值
		cache - 前向传播中对(Z, A)的缓存
		 X- dataset Y-Labels
	输出:
		delta_paras: ((dWm, dbm) ... (dW2, db2), (dW1, db1))
	"""

	delta_paras = []
	for i in range(layers): #总共m=4层
		W = paras['W'+str(layers-i)]
		b = paras['b'+str(layers-i)]
		Act = paras['F'+str(layers-i)]
		if( i == 0 ): #最后一层
			dA = cache[layers-1][1] - Y
		else:
			dA = last_dA #本次loop时，本层的dA等于上一次loop返回的last_dA
		if(i == layers - 1):
			last_A = X
		else:
			last_A = cache[layers-i-2][1]	

		Z = cache[layers - 1 - i][0]
		dZ, dW, db, last_dA = one_layer_backward(Z, W, b, dA, last_A, Act)
		T = (dW, db)
		delta_paras.append(T)
		"""
		print("\n\n\ndZ[" + str(layers-i) + "]: " + str(dZ))
		print("\ndA[" + str(layers-i-1) + "]: " + str(last_dA))
		print("\ndW[" + str(layers-i) + "]: " + str(dW))
		print("\ndb[" + str(layers-i) + "]: " + str(db))
		"""
	return delta_paras

def all_layers_forward(X, parameters, total_layers):
	"""
	输入:	X - 测试样本
		parameters: initialize_parameters的返回值
	输出: cache :{ "Z1" : Z1,
					"A1" : A1,
					"Z2" : Z2,
					"A2" : A2
						... }
	"""
	cache = []
	A = X
	for i in range(total_layers):
		index = i + 1
		W = parameters['W' + str(index)]
		b = parameters['b' + str(index)]
		f = parameters['F' + str(index)]
		last_A = A
		print ("all前向传播中的last_A: " + str (last_A))
		#print ("parameters: W: " + str(W) + " \n b: " + str(b) + " \n last_A: " + str(last_A) + " \n f" + str(f))
		Z, A = one_layer_forward(W, b, last_A, f)
		val = (Z, A)
		cache.append(val)
		#print ("第" + str(index) + "层的数据输出如下: \nZ:" + str(Z) +  "\nA:" + str(A))
	#print(cache)
	return cache
	
	

def initialize_parameters(): #返回固定
	#总计4层, 每一层节点数如下:
	n1 = 4
	n2 = 5
	n3 = 2
	n4 = 1

	#根据每一层节点数，确定w的矩阵大小，如下：
	W1 = np.random.random((n1, 3)) * 2 - 1
	W2 = np.random.random((n2, n1)) * 2 - 1
	W3 = np.random.random((n3, n2)) * 2 - 1
	W4 = np.random.random((n4, n3)) * 2 - 1
	"""
	W1 = np.random.random((n1, 3))
	W2 = np.random.random((n2, n1))
	W3 = np.random.random((n3, n2))
	W4 = np.random.random((n4, n3))
	"""

	b1 = b2 = b3 = b4 = 0
	
	parameters = { "W1": W1,
			"b1": b1,
			"F1" : "sigmoid",
			"W2": W2,
			"b2": b2,
			"F2" : "sigmoid",
			"W3": W3,
			"b3": b3,
			"F3" : "sigmoid",
			"W4": W4,
			"b4": b4,
			"F4" : "sigmoid" }
	return parameters

def update_paras(paras, delta_paras, learningRate):
	"""
	delta_paras: ((dWm, dbm) ... (dW2, db2), (dW1, db1))
	"""
	layers = len(delta_paras) #数组的长度等于层数
	for i in range(layers):
		W = paras['W'+str(layers -i)]
		#print ("in update,W: " + str(W))
		dW = delta_paras[i][0]
		#print ("in update,dW: " + str(dW))
		#print(W.shape)
		#print(dW.shape)
		assert(W.shape == dW.shape)
		paras['W'+str(layers - i)] -=  delta_paras[i][0] * learningRate
		paras['b'+str(layers - i)] -=  delta_paras[i][1] * learningRate

def model(X, Y, iteration_num, learningRate):
	paras = initialize_parameters()
	print("original parameters: " + str(paras))
	for i in range(iteration_num):
		cache = all_layers_forward(X, paras, 4) #cache : (Z, A)
		cost = compute_cost(cache[3][1], Y)
		#if( i % (iteration_num/10) == 0):
		#	print ("cost: " + str(cost))
		delta_paras = all_layers_backward(paras, 4, cache, X, Y) #做反向传播的时候并不会改变paras的值
		update_paras(paras, delta_paras, learningRate) #此处并不需要返回值，因为参数paras已经被修改，此处形参为地址,会改变params的值

	print("final parameters: " + str(paras))
	return paras

def predictions(paras, X, Y):
	"""
	X - 对给定的数据集做一次预测,数据集可能是训练集，也可能是测试集
	Y - labels
	paras - 做了n次迭代后获得更新的w，b参数的集合(调用update_paras后的值)
	"""
	cache = all_layers_forward(X, paras, 4)
	cost = compute_cost(cache[3][1], Y)
	print ("predictions result: " + str(cache[3][1]))
	print ("predictions-cost: " + str(cost))
	l2_error = cache[3][1] - Y
	print("error value: "+str(np.mean(np.abs(l2_error))))

	print ("\nFinal Z, A: \n" + str(cache))
	

#5个测试样本，每个样本3个特征(节点)值以便测试使用
X = np.array([[0, 1, 0, 0, 1],
	      [0, 1, 0, 0, 1],
	      [0, 1, 0, 0, 1]])
Y = np.array([[0, 1, 0, 0, 1]])

#paras = model(X, Y, 348, 0.01)
paras = model(X, Y, 1000, 0.01)
predictions(paras, X, Y)

"""
paras = initialize_parameters()
cache = all_layers_forward(X, paras, 4) #cache : (Z, A)
cost = compute_cost(cache[3][1], Y)
print ("cost: " + str(cost))

delta_paras = all_layers_backward(paras, 4, cache, X, Y)
learningRate = 0.01
print ("Before update: " + str(paras))
new_paras = update_paras(paras, delta_paras, learningRate)
print ("After update: " + str(paras))

#第4层到第3层的反向传播
w4 = paras["W4"]
b4 = paras["b4"]
Act4 = paras["F4"]
dA4 = cache[3][1] - Y
A3 = cache[2][1]

dZ4, dw4, db4, dA3 = one_layer_backward(w4, b4, dA4, A3, Act4)
print("dZ4: " + str(dZ4))
print("\ndA3: " + str(dA3))
print("\ndw4: " + str(dw4))
print("\ndb4: " + str(db4))

#第3层到第2层的反向传播
w3 = paras["W3"]
b3 = paras["b3"]
Act3 = paras["F3"]
A2 = cache[1][1]

dZ3, dw3, db3, dA2 = one_layer_backward(w3, b3, dA3, A2, Act3)
print("dZ3: " + str(dZ3))
print("\ndA2: " + str(dA2))
print("\ndw3: " + str(dw3))
print("\ndb3: " + str(db3))

#第2层到第1层的反向传播
w2 = paras["W2"]
b2 = paras["b2"]
Act2 = paras["F2"]
A1 = cache[0][1]

dZ2, dw2, db2, dA1 = one_layer_backward(w2, b2, dA2, A1, Act2)
print("dZ2: " + str(dZ2))
print("\ndA1: " + str(dA1))
print("\ndw2: " + str(dw2))
print("\ndb2: " + str(db2))

#第1层到第0层的反向传播
w1 = paras["W1"]
b1 = paras["b1"]
Act1 = paras["F1"]
A0 = X

dZ1, dw1, db1, dA0 = one_layer_backward(w1, b1, dA1, A0, Act1)
print("dZ1: " + str(dZ1))
print("\ndA0: " + str(dA0))
print("\ndw1: " + str(dw1))
print("\ndb1: " + str(db1))
"""
