#coding=utf-8

import numpy as np
import bruce

def dictionary_to_vector(parameters):
	"""
	将权重参数parameters, {"W1":W1, "b1":b1, "W2":W2, "b2":b2 ...}转化成(n, 1)的vector
	输入: 权重参数
	输出: 由权重参数堆积而成的n行，1列向量
	"""
	keys = []
	count = 0
	L = len(parameters)//2
	for i in range(0, L):
		for key in ["W"+str(i+1), "b"+str(i+1)]:
        
			# flatten parameter
			new_vector = np.reshape(parameters[key], (-1,1))
			#keys = keys + [key]*new_vector.shape[0]

			if count == 0:
				theta = new_vector
			else:
				theta = np.concatenate((theta, new_vector), axis=0)
			count = count + 1

	return theta

def gradients_to_vector(gradients):
	"""
	功能:
		将梯度参数gradients, {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2 ...}转化成(n, 1)的vector
	输入: 
		权重参数
	输出: 
		由权重参数堆积而成的n行，1列向量
	"""
	keys = []
	count = 0
	L = len(gradients)//3 #在反向传播过程中，返回值包含了dA, dW, db, 故长度应该除以3
	for i in range(0, L):
		for key in ["dW"+str(i+1), "db"+str(i+1)]:
        
			# flatten parameter
			new_vector = np.reshape(gradients[key], (-1,1))
			#keys = keys + [key]*new_vector.shape[0]

			if count == 0:
				theta = new_vector
			else:
				theta = np.concatenate((theta, new_vector), axis=0)
			count = count + 1

	return theta

def vector_to_dictionary(thetaminus, layers_dims):
	"""
	功能:将列向量转换回权重参数矩阵
	输入：
		thetaminus - 权重参数的列向量
		layers_dims - 神经网络各层节点数(特征数) e.g (12888, 20, 7, 5, 1), 用于复原权重参数矩阵
	输出:
		权重参数W,b的矩阵
	"""

	#print ("thetaminus: " + str(thetaminus))
	L = len(layers_dims) - 1;
	parameters = {}
	beg = 0
	end = 0
	for i in range(L):
		beg = end
		step = layers_dims[i+1] * layers_dims[i]
		end = beg + step
		parameters["W"+str(i+1)] = thetaminus[beg:end].reshape((layers_dims[i+1], layers_dims[i]))

		beg = end
		step = layers_dims[i+1]
		end = beg + step
		parameters["b"+str(i+1)] = thetaminus[beg:end].reshape((layers_dims[i+1], 1))

	return parameters

def gradient_check(parameters,gradients,X,Y,epsilon=1e-7, layers_dims=(10, 5, 1)):
	"""
	检查backward_propagation_n是否正确计算forward_propagation_n输出的成本梯度

	#梯度检查原理:
		1, 将最后获得的权重参数,针对所有参数，依次增加一个很小的值(epsilon),执行一次前向传播，然后得到cost值J_plus
		2, 针对所有参数减去一个很小的值(epsilon),执行一次前向传播，然后得到cost值J_minus
		3, 根据公式 gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon) 计算出双边梯度

		4, 计算单边梯度(反向传播梯度)与双边梯度的差的2范数
		5, 计算单边梯度与双边梯度的2范数之和
		6, 计算4，5之间的比例，即为可容忍的梯度

	参数：
		parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
		grad_output_propagation_n的输出包含与参数相关的成本梯度。
		x  - 输入数据点，维度为（输入节点数量，1）
		y  - 标签
		epsilon  - 计算输入的微小偏移以计算近似梯度
		layers_dims - 神经网络各层节点数(特征数) e.g (12888, 20, 7, 5, 1), 用于计算权重参数W,b向量化后转换回权重格式，以用于前向传播

	返回：
		difference - 近似梯度和后向传播梯度之间的差异
	"""
	#初始化参数
	layers_dims = (X.shape[0], 5, 3, 1) #hardcode for testing only
	assert (len(parameters)//2 == len(layers_dims) - 1)
	parameters_values = dictionary_to_vector(parameters)
	grad = gradients_to_vector(gradients)
	assert (parameters_values.shape == grad.shape)
	num_parameters = parameters_values.shape[0]
	J_plus = np.zeros((num_parameters,1))
	J_minus = np.zeros((num_parameters,1))
	gradapprox = np.zeros((num_parameters,1))

	#计算gradapprox
	for i in range(num_parameters):
		#计算J_plus [i]。输入：“parameters_values，epsilon”。输出=“J_plus [i]”
		thetaplus = np.copy(parameters_values)                                                  # Step 1
		thetaplus[i][0] = thetaplus[i][0] + epsilon                                             # Step 2
		parameters = vector_to_dictionary(thetaplus, layers_dims)
		#print ("vector to dic: " + str(parameters))
		AL , caches = bruce.L_model_forward(X,parameters, 1) #keep_probs = 1, caches 用不到
		J_plus[i] = bruce.compute_cost(AL,Y) #未计算L2正则惩罚项

		#计算J_minus [i]。输入：“parameters_values，epsilon”。输出=“J_minus [i]”。
		thetaminus = np.copy(parameters_values)                                                 # Step 1
		thetaminus[i][0] = thetaminus[i][0] - epsilon                                           # Step 2        
		AL , caches = bruce.L_model_forward(X,vector_to_dictionary(thetaminus, layers_dims), 1) #keep_probs = 1, caches 用不到
		J_minus[i] = bruce.compute_cost(AL,Y) #未计算L2正则惩罚项

		#计算gradapprox[i]
		gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

	#通过计算差异比较gradapprox和后向传播梯度。
	numerator = np.linalg.norm(grad - gradapprox)                                     # Step 1'
	denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                   # Step 2'
	difference = numerator / denominator                                              # Step 3'

	if difference < 1e-7:
		print("梯度检查：梯度正常! difference= " + str(difference))
	else:
		print("梯度检查：梯度超出阈值! difference= " + str(difference))

	return difference
