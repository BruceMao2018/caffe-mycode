#coding=utf-8
import numpy as np
from load_data import load_dataset
np.random.seed(1) #指定随机种子，这样的话每次随机的时候取值就是一样的，便于比较

"""
Implement the cost function and its gradient for the propagation explained above


Return:
cost -- negative log-likelihood cost for logistic regression
dw -- gradient of the loss with respect to w, thus same shape as w
db -- gradient of the loss with respect to b, thus same shape as b

Tips:
- Write your code step by step for the propagation. np.log(), np.dot()
"""
def sigmoid(x, Deriv = False):
	if( Deriv == False): #前向传播
		return 1/(1+np.exp(-x))
	return x * (1 -x) #反向传播

def propagate(w, b, X, Y):
	"""
	Exercise: Implement a function propagate() that computes the cost function and its gradient.
	Hints:
	Forward Propagation: 
	- You get X 
	- You compute [Math Processing Error]A=σ(wTX+b)=(a(0),a(1),...,a(m−1),a(m)) 
	- You calculate the cost function: [Math Processing Error]J=−1m∑i=1my(i)log⁡(a(i))+(1−y(i))log⁡(1−a(i))

	Arguments:
	w -- weights, a numpy array of size (num_px * num_px * 3, 1)
	b -- bias, a scalar
	X -- data of size (num_px * num_px * 3, number of examples)
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
	"""
	m =  X.shape[1]
	#print ("样本数: " +  str(m))
	Z = np.dot(w.T, X) + b
	A = sigmoid(Z)
	cost = -(1.0/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
	dw = (1.0/m) * np.dot(X, (A-Y).T)
	db = (1.0/m) * np.sum(A - Y)
	#print ("w.shape: " + str(w.shape))
	print ("dw: " + str(dw))
	#print ("cost: " + str(cost))
	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	grads = {"dw": dw,
		"db": db}
	return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	costs = []
	for i in range(num_iterations):
		#前向传播
		grads, cost = propagate(w, b, X, Y)

		dw = grads["dw"]
		db = grads["db"]

		#更新w, b
		w = w - learning_rate*dw
		b = b - learning_rate*db

        	# Record the costs
		if i % (num_iterations/10) == 0:
			costs.append(cost)

		# Print the cost every 100 training examples
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))

		params = {"w": w,
			"b": b}

		grads = {"dw": dw,
			"db": db}

	return params, grads, costs

def predict(w, b, X): #对所有样本做一次预测
	'''
	Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

	Arguments:
	w -- weights, a numpy array of size (num_px * num_px * 3, 1)
	b -- bias, a scalar
	X -- data of size (num_px * num_px * 3, number of examples)

	Returns:
	Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
	'''

	m = X.shape[1] #获得测试样本个数
	Y_prediction = np.zeros((1,m)) #初始化预测值
	w = w.reshape(X.shape[0], 1)

	# Compute vector "A" predicting the probabilities of a cat being present in the picture
	### START CODE HERE ### (≈ 1 line of code)
	A = sigmoid(np.dot(w.T, X) + b)
	### END CODE HERE ###
	print ("预测: " + str(A))

	for i in range(A.shape[1]):
		if A[0,i] > 0.5:
			Y_prediction[0,i] = 1
		else:
			Y_prediction[0,i] = 0

	assert(Y_prediction.shape == (1, m))

	return Y_prediction

"""
X = np.array([[1, 2],
		[2, 4],
		[3, 6],
		[4, 8],
		[5, 10],
		[6, 12]])

print ("输入样本X: " + str(X.shape))

w = np.random.random((6, 1))
Y = np.array([[0, 1]])
"""

"""
X = np.array([[0,0,1],
		[0,1,1],
		[1,0,1],
		[1,1,1],
		[0,0,1]])
X = X.T #(3, 5)
print (X.shape) #(5,3)-意味着一共有5个样本，每个样本有3个特征

Y = np.array([[0],
			 [0],
			 [0],
			 [1],
			 [0]])
Y = Y.T #(1, 5)

w = np.random.random((3, 1))

print ("X.size: " + str(X.shape))
print ("Y.size: " + str(Y.shape))
print ("w.size: " + str(w.shape))
b = 0.01
params, grads, cost = optimize(w, b, X, Y, 10000, 0.01, False)

print ("params:" + str(params))
print ("dw:" + str(grads))
print ("cost: " + str(cost))
print ("w: " + str(params["w"]))
print ("b: " + str(params["b"]))

Y_prediction = predict(w, b, X)	#对所有样本做一次预测
print ("对所有样本做一次预测: " + str(Y_prediction))
"""
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
print ("训练集大小: " + str(train_set_x_orig.shape))
print ("训练集标签大小: " + str(train_set_y_orig.shape))
print ("测试集大小: " + str(test_set_x_orig.shape))
print ("测试集标签大小: " + str(test_set_y_orig.shape))
print ("分类：" + str(classes))

X = train_set_x_orig
X = X.reshape(X.shape[1] * X.shape[2] * X.shape[3], X.shape[0])
Y = train_set_y_orig

print ("input data: X - " + str(X.shape) + " labels: " + str(Y.shape))

w = np.random.random((X.shape[0], 1))

print ("X.size: " + str(X.shape))
print ("Y.size: " + str(Y.shape))
print ("w.size: " + str(w.shape))
b = 0.01
params, grads, costs = optimize(w, b, X, Y, 100, b, print_cost = False)

print ("params:" + str(params))
print ("w: " + str(params["w"]))
print ("b: " + str(params["b"]))

Y_prediction = predict(w, b, X)	#对所有样本做一次预测
print ("对所有样本做一次预测: " + str(Y_prediction))
print ("训练集标签: " + str(Y))
