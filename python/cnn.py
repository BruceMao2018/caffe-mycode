#coding=utf-8
#		L0 -w0	L1 -w1	L2	L3(output)
#		___		___	___	___
#		| |	 _ 	| |	| |	|_|
#	x1	|_|	| |	|_|S	|_|R	|_|
#	 	| |	|_|S	| |i	| |e	|_|
#	x2	|_|	| |i	|_|g	|_|L	|_|
#	 	| |	|_|g	| |m	| |u	|_|
#	x3	|_|	| |m	|_|o	|_|	|_|
#	 	| |	|_|o	| |i	| |	|_|
#	x4	|_|	| |i	|_|d	|_|	|_|
#	 	| |	|_|d	| |	|_|	|_|
#	x5	|_|	   	|_|	|_|	|_|
#
#
#	x	|_|	|_|	|_|	|_|	|_|
#coding=utf-8
import numpy as np
def sigmoid(x, Deriv=False):
	if(Deriv == True): #求导操作
		return x * (1 -x)
	return 1/(1+np.exp(-x))

def ReLu(x, Deriv=False):
	if(Deriv == True):
		return x>0
	else:
		return x*(x>0)

x = np.array([[0,0,1,0],
		[0,1,0,1],
		[1,1,1,1],
		[0,0,1,0],
		[1,0,0,1]])
print (x.shape)

y = np.array([[0],[0],[1],[0],[0]])
print (y.shape)

np.random.seed(1) #指定随机种子，这样的话每次随机的时候取值就是一样的，便于比较

w0 = 2 * np.random.random((4,4)) - 1
w1 = 2 * np.random.random((4,5)) - 1
w2 = 2 * np.random.random((5,1)) - 1

#开始前向传播
num = 5000
tmp_num = num/5
print tmp_num
for i in xrange(num):
	l0 = x
	l1 = np.dot(l0, w0)
	l1 = sigmoid(l1)
	#l1 = ReLu(l1, False)

	l2 = np.dot(l1, w1)
	l2 = sigmoid(l2)
	#l2 = ReLu(l2, False)

	l3 = np.dot(l2, w2)
	l3 = sigmoid(l3, False)
	#l3 = ReLu(l3, False)


#反向传播
	l3_error = y - l3
	if( i%tmp_num == 0):
		print("error value: "+str(np.mean(np.abs(l3_error))))
	l3_delta = l3_error * sigmoid(l3, True)
	#l3_delta = l3_error * ReLu(l3, True)
	#l3 = l3 - l3_delta

	l2_error = np.dot(l3_delta, w2.T)
	l2_delta = l2_error * sigmoid(l2, True)
	#l2_delta = l2_error * ReLu(l2, True)
	#l2 = l2 - l2_delta

	l1_error = np.dot(l2_delta, w1.T)
	l1_delta = l1_error * sigmoid(l1, True)
	#l1_delta = l1_error * ReLu(l1, True)
	#l1 = l1 - l1_delta

	w2_delta = np.dot(l2.T, l3_delta)
	w1_delta = np.dot(l1.T, l2_delta)
	w0_delta = np.dot(l0.T, l1_delta)
	w2 += w2_delta
	w1 += w1_delta
	w0 += w0_delta
