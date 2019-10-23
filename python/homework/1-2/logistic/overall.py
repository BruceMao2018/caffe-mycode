#coding=utf-8
import numpy as np
from load_data import load_dataset

np.random.seed(1) #指定随机种子，这样的话每次随机的时候取值就是一样的，便于比较

#先定义激活函数及激活函数的导数
def sigmoid(x, Deriv = False):
	if( Deriv == False): #前向传播
		return 1/(1+np.exp(-x))
	return x * (1 -x) #反向传播

def ReLu(x, Driver = False):
	if( Driver == True): #反向传播
		return (x > 0)
	return x * (x > 0) #前向传播

#定义softmax分类器,softmax分类器是针对输出向量进行多类别分类，然后返回一个概率向量
#输入x的矩阵为(n, m), n表示分类的类别数量, m表示样本个数
def softmax(x):
	exp_x = np.exp(x)
	s = np.sum(exp_x, axis = 0, keepdims = True)
	return exp_x/s;

def optimize_1(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	"""
定义前向、反向传播
输入值:		X - Train dataset
		Y - train labels
		w - 权重参数
矩阵大小:
		X - (nx, m) nx - height * width * Channesl
		Y - (1, m) m : 样本个数
		w - (1, nx) 此softmax只是作为最好一层的激活函数，最后一层的Y为(1, m), 因为 Y = W.X, 故w的大小为(1, nx)
		b - (1, m) 矩阵大小等同于输出-Y
返回新的参数w, b
	"""
	for i in range(num_iterations):
		#前向传播
		Z = np.dot(w.T, X) + b
		A = sigmoid(Z)

		#反向传播
		m = Y.shape[1]
		dw = (1.0/m) * np.dot(X, (A-Y).T)
		db = (1.0/m) * np.sum(A - Y)

		assert(dw.shape == w.shape)
		assert(db.dtype == float)

		cost = -(1.0/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
		cost = np.squeeze(cost)
		assert(cost.shape == ())
		"""
		if(i % (num_iterations/10) == 0):
			print ("dw: " + str(dw))
			print ("db: " + str(db))
			print ("cost: " + str(cost))
		"""

		#更新参数
		#w = w - learning_rate*dw
		#b = b - learning_rate*db
		w = w - dw * learning_rate
		b = b - db * learning_rate

		#print ("dw: " + str(dw))
		#print ("db: " + str(db))
		#print ("cost: " + str(cost))

	params = {"w": w,
		"b": b}

	return params



def optimize_2(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	"""
定义前向、反向传播
输入值:		X - Train dataset
		Y - train labels
		w - 权重参数
矩阵大小:
		X - (nx, m) nx - height * width * Channesl
		Y - (1, m) m : 样本个数
		w - (1, nx) 此softmax只是作为最好一层的激活函数，最后一层的Y为(1, m), 因为 Y = W.X, 故w的大小为(1, nx)
		b - (1, m) 矩阵大小等同于输出-Y
返回新的参数w, b
	"""
	for i in range(num_iterations):
		#前向传播
		Z = np.dot(w, X) + b
		A = sigmoid(Z)

		#反向传播
		m = Y.shape[1]
		dA = A - Y
		dZ = dA * sigmoid(Z, Deriv = True)

		dw = np.dot(dZ, X.T)
		#db = dZ * 0.1
		db = 0.0

		assert(dw.shape == w.shape)
		#assert(db.dtype == float)

		cost = (1.0/m) * np.sum(np.power((A - Y), 2))

		"""
		if(i % (num_iterations/50) == 0):
			print ("dw: " + str(dw))
			print ("db: " + str(db))
			print ("cost: " + str(cost))
		"""

		#更新参数
		#w = w - learning_rate*dw
		#b = b - learning_rate*db
		w = w - dw * learning_rate
		b = b - db * learning_rate

		#print ("dw: " + str(dw))
		#print ("db: " + str(db))
		#print ("cost: " + str(cost))

	params = {"w": w,
		"b": b}

	return params

"""
根据一定次的迭代后返回的更新参数w,b对测试集做一次预测
Returns:
	Y_prediction -- 返回结果为所有测试样本的分类结果,本列使用二分类,返回的结果是(1, m), 表示是否为cat
"""
def predict(w, b, X):
	Z = np.dot(w.T, X) + b
	A = sigmoid(Z)
	m = X.shape[1]
	Y_prediction = np.zeros((1,m)) #初始化预测值

	for i in range(m):
		if( A[0, i] > 0.5 ):
			Y_prediction[0, i] = 1
		else:
			Y_prediction[0, i] = 0

	#print (Y_prediction)
	return Y_prediction

"""
X = np.array([[0, 0, 1, 1, 0],
		[0, 1, 0, 1, 0],
		[1, 1, 1, 1, 1]])
print (X.shape) #(3,5)-意味着一共有5个样本，每个样本有3个特征

Y = np.array([[0, 0, 0, 1, 0]])
w = np.random.random((1, 3))
"""

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
print ("训练集大小: " + str(train_set_x_orig.shape))
print ("训练集标签大小: " + str(train_set_y_orig.shape))
print ("测试集大小: " + str(test_set_x_orig.shape))
print ("测试集标签大小: " + str(test_set_y_orig.shape))
print ("分类：" + str(classes))

X = train_set_x_orig
X = X.reshape(X.shape[1] * X.shape[2] * X.shape[3], X.shape[0])
X = X/255; #一定要做归一化操作!!!!!
Y = train_set_y_orig

print ("input data: X - " + str(X.shape) + " labels: " + str(Y.shape))

#w = np.random.random((X.shape[0], 1))
w = np.zeros((X.shape[0], 1))

print ("X.size: " + str(X.shape))
print ("Y.size: " + str(Y.shape))
print ("w.size: " + str(w.shape))
b = 0.01
params = optimize_1(w, b, X, Y, 2000, b, False)
w_new_1 = params["w"]
b_new_1 = params["b"]
print ("w_new_1: " + str(w_new_1))
Y_prediction = predict(w_new_1, b_new_1, X)	#对所有样本做一次预测
print ("对所有样本做第一次预测: " + str(Y_prediction))
print("train accuracy1: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))

print ("params:" + str(params))
print ("w: " + str(params["w"]))
print ("b: " + str(params["b"]))

w = 2 * np.random.random((1, X.shape[0])) - 1
b = 0
params = optimize_2(w, b, X, Y, 2000, b, False)
w_new_2 = params["w"]
b_new_2 = params["b"]
print ("w_new_2: " + str(w_new_2))
Y_prediction = predict(w_new_2.T, b_new_2, X)	#对所有样本做一次预测
print ("对所有样本做第二次预测: " + str(Y_prediction))
print("train accuracy2: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))
print ("训练集标签: " + str(Y))

# Print train/test Errors
#print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))
#print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
